from beesup_llm import get_labhandler, _isinstance
from beesup_llm.llm_evaluation import LLMEvaluator
from beesup_llm.llm import LLMPipeline, prepare_sample_for_chat_completion, prepare_sample_for_chat_finetuning

import logging
import pandas as pd

import re
import math
import yaml
import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback, TrainingArguments, DataCollatorForSeq2Seq
from transformers.trainer import TrainerState
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training


import pytz
import datetime
TIMEZONE = pytz.timezone('Europe/Berlin')
TIMESTAMP_FORMAT='%Y-%m-%d_%H-%M-%S'

def get_datetime(a_timestamp=None):
    if a_timestamp is not None:
        return datetime.datetime.strptime(a_timestamp, TIMESTAMP_FORMAT).replace(tzinfo=TIMEZONE)
    else:
        return datetime.datetime.now(TIMEZONE)

def get_timestamp(a_datetime=None):
    if a_datetime is None:
        a_datetime=get_datetime()

    return a_datetime.strftime(TIMESTAMP_FORMAT)

def camel_to_snake(name: str) -> str:
    # Insert underscores before capital letters and convert to lowercase
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


class EvaluatorCallback(TrainerCallback):

    def __init__(self, experiment, evaluator: LLMEvaluator, **kwargs):

        self.experiment=experiment
        self.evaluator=evaluator
        self.output_dir=getattr(self.experiment,'sft_config',{}).get('output_dir','.')
    
    @property
    def snake_name(self):
        return camel_to_snake(self.__class__.__name__)
    
    def get_df_name(self, state: TrainerState) -> str:
        return f"{self.snake_name}_{state.global_step}_{self.output_name}_df.pkl"
    
    def do_log(self, state: TrainerState) -> None:
        self.experiment.logger.info(f"epoch {state.epoch}/ global_step {state.global_step} --> {self.get_df_name(state)}")	

    def save_df(self, df: pd.DataFrame, state: TrainerState) -> None:
        df_name=self.get_df_name(state)
        df_path=f"{self.output_dir}/{df_name}"
        df.to_pickle(df_path)
        self.experiment.logger.info(f"Saved results to {df_path}")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        return 

class LogHistoryCallback(TrainerCallback):
    def __init__(self, experiment):
        self.experiment=experiment

    def on_epoch_end(self, args=None, state=None, control=None, **kwargs):
        log_history_df=pd.DataFrame(state.log_history)
        log_history_df.to_pickle(f"{self.experiment._path}/log_history_df.pkl")

class CustomSFTTrainer(SFTTrainer):
    """Custom SFTTrainer that stores per-sample losses during training."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Get the usual loss and outputs from the parent
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        global_step=int(self.state.global_step)+1
        epoch=int(self.state.epoch+global_step/self.state.max_steps)

        if "labels" in inputs:
            logits = outputs.logits
            logits = logits[..., :-1, :].contiguous() # Shift so that tokens < n predict n
            logits = logits.permute(0,2,1) # (batch_size, vocab_size, seq_len)

            labels = inputs["labels"]  # (batch_size, seq_len)
            labels = labels[..., 1:].contiguous() # Shift so that tokens < n predict n


            loss_fct = torch.nn.CrossEntropyLoss()
            tokens_per_sample=torch.tensor([(labs != -100).sum().item() for labs in labels])
            batch_proportions=tokens_per_sample/tokens_per_sample.sum()
            batch_proportions=batch_proportions.tolist()
            loss_per_sample=torch.tensor([loss_fct(logs.unsqueeze(0), labs.unsqueeze(0)).item() for logs, labs in zip(logits, labels)])
            loss_per_sample=loss_per_sample.detach().cpu().tolist()

            if "sample_ids" in inputs:
                sample_ids = inputs["sample_ids"].cpu().tolist()
            else:
                sample_ids = labels.size(0)*[0]  # fallback
            
            loss_data=[]
            for l, s, p in zip(loss_per_sample, sample_ids, batch_proportions):
                loss_data.append(
                    dict(
                        global_step=global_step,
                        epoch=epoch,
                        sample_id=s,
                        loss=l,
                        batch_proportion=p,
                    )
                )

            # Now "push" these losses to any callback that implements `add_loss`
            for callback in self.callback_handler.callbacks:
                if hasattr(callback, "add_loss_data"):
                    callback.add_loss_data(loss_data)

        return (loss, outputs) if return_outputs else loss

class CustomDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    """Custom DataCollatorForSeq2Seq that returns sample IDs in the batch."""
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        # Extract sample IDs before processing
        sample_ids = [feature["sample_id"] for feature in features if "sample_id" in feature]

        # Call the original DataCollatorForSeq2Seq method
        batch = super().__call__(features, return_tensors=return_tensors)

        # Add sample_ids back to the batch
        if sample_ids:
            batch["sample_ids"] = torch.tensor(sample_ids, dtype=torch.long)

        return batch


class FinetuningExperiment(object):
    """
    Performs supervised fine-tuning (SFT) of a LLM applying Low-Rank-Adaptation (LoRA).
    """
    logger=logging.getLogger(__name__)

    @classmethod
    def fit_llm_pipe(cls, llm_pipe: LLMPipeline, **kwargs) -> LLMPipeline:
        llm_pipe.generation_config['max_new_tokens']=4096
        llm_pipe.generation_config['max_time']=1200
        return llm_pipe

    def __init__(
            self,
            ref=None,
            label:str=None,
            llm_pipe: LLMPipeline=None,
            data_df=None, # Dataframe with column 'split' for train/eval
            evaluators:list=[],
            labh=get_labhandler(),
            **kwargs):
        
        self.label=label
        self.done=False
        
        self.do_eval_base_model=kwargs.get('do_eval_base_model',True)
        self.do_eval_lora_model=kwargs.get('do_eval_lora_model',True)
        self.do_train=kwargs.get('do_train',True)
    
        #LORA CONFIG
        self.lora_config=kwargs.get('lora_config',dict(
            r=kwargs.get('r',32),
            lora_alpha=kwargs.get('lora_alpha',3),
            use_rslora=kwargs.get('use_rslora',True),
            target_modules=kwargs.get('target_modules','all-linear'),
            lora_dropout=kwargs.get('lora_dropout',0.05),
            bias=kwargs.get('bias','none'),
            task_type=kwargs.get('task_type','CAUSAL_L')
        ))

        #SUPERVISED FINE-TUNING TRAINER CONFIG
        self.sft_config=kwargs.get('sft_config',dict(
            num_train_epochs=kwargs.get('num_train_epochs',10),
            output_dir=kwargs.get('output_dir','.'),
            auto_find_batch_size=kwargs.get('auto_find_batch_size',True),
            per_device_train_batch_size=kwargs.get('per_device_train_batch_size',8),
            gradient_accumulation_steps=kwargs.get('gradient_accumulation_steps',1),
            learning_rate=kwargs.get('learning_rate',0.0002),
            optim=kwargs.get('optim','paged_adamw_8bit'),
            save_strategy=kwargs.get('save_strategy','no'),
            eval_strategy=kwargs.get('eval_strategy','no'),
            logging_strategy=kwargs.get('logging_strategy','steps'),
            logging_steps=kwargs.get('logging_steps',1),
            logging_first_step=kwargs.get('logging_first_step',True),
            do_train=kwargs.get('do_train',True),
            do_eval=kwargs.get('do_eval',False),
            report_to=kwargs.get('report_to','none'),
            max_seq_length=kwargs.get('max_seq_length',4096),
            packing=kwargs.get('packing',False),
            seed=kwargs.get('seed',42)
        ))

        if labh is not None:
            self.labh=labh(locals())
            data_df=self.labh.handle_object(locals(),'data_df')
            llm_pipe=self.labh.handle_object(locals(),'llm_pipe')
            evaluators=self.labh.handle_object(locals(),'evaluators')
            self.sft_config['output_dir']=self._path

            #if self.is_saved:

        if isinstance(data_df, pd.DataFrame):
            self.data_df = data_df.copy(); del data_df
        
        if _isinstance(llm_pipe, LLMPipeline):
            llm_pipe = self.fit_llm_pipe(llm_pipe)
            self.llm_pipe = llm_pipe

        if isinstance(evaluators, list) and all([_isinstance(e, LLMEvaluator) for e in evaluators]):
            self.evaluators=evaluators
    
    def load_data(self, **kwargs) -> None:

        assert hasattr(self, 'data_df'), "data_df is missing"
        assert 'split' in self.data_df.columns, "split column is missing"
        assert hasattr(self, 'llm_pipe'), "llm_pipe is missing"

        train_df=self.data_df[self.data_df['split']=='train'].reset_index(drop=True).copy()
        eval_df=self.data_df[self.data_df['split']=='eval'].reset_index(drop=True).copy()

        self.train_ds, self.eval_ds = None, None

        if not train_df.empty:
            self.llm_pipe.load_tokenizer('training')
            self.train_ds=Dataset.from_list(train_df.apply(lambda x: prepare_sample_for_chat_finetuning(x, self.llm_pipe.get_tokenizer('training'),**kwargs), axis=1).to_list())
        
        if not eval_df.empty:
            self.llm_pipe.load_tokenizer('inference')
            self.eval_ds=Dataset.from_list(eval_df.apply(lambda x: prepare_sample_for_chat_completion(x, self.llm_pipe.get_tokenizer('inference'),**kwargs), axis=1).to_list())
        
        return
    
    def get_lora_model(self, **kwargs) -> PeftModel:
        self.logger.info(f"Loading Lora model")

        model=self.llm_pipe.get_model()
        lora_config=LoraConfig(**self.lora_config)

        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.config.use_cache = False
        model.print_trainable_parameters()

        #SET LORA INFO
        n_trainable_params,n_total_params=model.get_nb_trainable_parameters()
        p_trainable_params=n_trainable_params/n_total_params

        if lora_config.use_rslora:
            lora_scale=lora_config.lora_alpha/math.sqrt(lora_config.r)
        else:
            lora_scale=lora_config.lora_alpha/lora_config.r

        self.lora_info={
            'n_trainable_params':n_trainable_params,
            'n_total_params':n_total_params,
            'p_trainable_params':p_trainable_params,
            'lora_scale':lora_scale,
        }

        if hasattr(self, 'save_config'): self.save_config()
        return model
    
    def load_trainer(self, **kwargs) -> None:
        self.logger.info(f"Loading trainer")

        lora_model=self.get_lora_model(**kwargs)

        data_collator=CustomDataCollatorForSeq2Seq(
            tokenizer=self.llm_pipe.get_tokenizer('training'),
            model=lora_model,
            padding='longest',
            label_pad_token_id =-100
            )
        
        self.trainer=CustomSFTTrainer(
            model=lora_model,
            data_collator=data_collator,
            train_dataset=self.train_ds,
            eval_dataset=self.eval_ds,
            peft_config=LoraConfig(**self.lora_config),
            args=SFTConfig(**self.sft_config),
            )
    
        return
    
    def evaluate_base_model(self, callback_class = EvaluatorCallback, **kwargs) -> None:
        self.logger.info(f"Evaluating base model")

        for evaluator in self.evaluators:
            evaluator_callback=callback_class(evaluator, self, **kwargs)
            evaluator_callback.on_epoch_end(state=TrainerState(epoch=0), model=self.llm_pipe.get_model(), **kwargs)

        return
    
    def train(self, **kwargs) -> None:

        with open(f"{self.sft_config['output_dir']}/trainer_args.yaml", 'w') as file:
            yaml.dump(self.trainer.args.to_dict(), file, sort_keys=False, default_flow_style=False)
        
        self.timestamp_train_start=get_timestamp()
        self.trainer.train()
        
        self.timestamp_train_done=get_timestamp()

        if hasattr(self, 'save_config'): self.save_config()
        return

    def run(self, **kwargs) -> None:

        self.timestamp_start=get_timestamp()
        self.llm_pipe.prepare_inference()
        self.load_data(**kwargs)
        
        if self.do_eval_base_model:
            self.evaluate_base_model(**kwargs)
        
        if (self.do_eval_lora_model or self.do_train):
            self.load_trainer(**kwargs)

        if self.do_eval_lora_model:
            for evaluator in self.evaluators:
                evaluator_callback=EvaluatorCallback(evaluator, self, **kwargs)
                self.trainer.add_callback(evaluator_callback)
        
        if self.do_train:
            self.train(**kwargs)

        self.done=True
        self.timestamp_done=get_timestamp()
        return


    
   


    

