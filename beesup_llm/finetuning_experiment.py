from beesup_llm import get_labhandler, _isinstance
from beesup_llm.evaluation import LLMEvaluator
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

def get_state_tag(state: TrainerState) -> str:
    return f"{int(state.epoch)}-{int(state.global_step)}"

class ExperimentCallback(TrainerCallback):

    logger=logging.getLogger(__name__)
    
    def __init__(self, experiment, **kwargs):
        self.experiment=experiment
        self.output_dir=getattr(self.experiment,'sft_config',{}).get('output_dir','.')

        self.class_tag=camel_to_snake(self.__class__.__name__)
        self.object_tag=str()

    def save_df(self, df: pd.DataFrame, state: TrainerState) -> None:
        self.state_tag=get_state_tag(state)

        file_path=f"{self.output_dir}/"
        file_name='_'.join([s for s in [self.state_tag, self.object_tag, self.class_tag, 'df.pkl'] if s])
        file_path+=file_name

        df.to_pickle(file_path)
        self.logger.debug(f"{self.state_tag} {self.object_tag} Saved results to '{file_path}'")

    def on_epoch_end(self, args=None, state=None, control=None, **kwargs):
        self.state_tag=get_state_tag(state)
        self.logger.info(f"Start Evaluation {self.class_tag} {self.object_tag} {self.state_tag}")
        return

class EvaluatorCallback(ExperimentCallback):

    def __init__(self, *args, evaluator: LLMEvaluator, **kwargs):
        super().__init__(*args, **kwargs)

        if _isinstance(evaluator, LLMEvaluator):
            self.evaluator=evaluator
    
class MCECallback(ExperimentCallback):
    """Multiclass Cross Entropy Loss Evaluator Callback,

    fetches sample-mapped loss data from Custom Trainer Wrapper
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_data=[]

    def add_loss_data(self, data: list) -> None:
        self.loss_data.extend(data)

    def on_epoch_end(self, args, state, control, **kwargs) -> None:
        super().on_epoch_end(args, state, control, **kwargs)

        callback_df=pd.DataFrame(self.loss_data)
        self.loss_data = [] #reset loss data

        self.save_df(callback_df, state)
        return

class HistoryCallback(ExperimentCallback):

    def on_epoch_end(self, args, state, control, **kwargs) -> None:
        super().on_epoch_end(args, state, control, **kwargs)

        callback_df=pd.DataFrame(state.log_history)
        self.save_df(callback_df, state)
        return


class CustomSFTTrainer(SFTTrainer):
    """Custom SFTTrainer that stores per-sample losses during training. Required when using a MCECallback."""
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
    """Custom DataCollatorForSeq2Seq that returns sample IDs in the batch. Required when using a MCECallback."""
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


class FinetuningExperiment:
    """
    This class is designed to perform supervised fine-tuning (SFT) of a large language model (LLM) 
    using Low-Rank Adaptation (LoRA). It provides methods for configuring, training, and evaluating 
    the model, as well as handling data preparation and LoRA-specific configurations.
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
        self.done=kwargs.get('done',False)

        self.do_eval_base_model=kwargs.get('do_eval_base_model',True)
        self.do_eval_lora_model=kwargs.get('do_eval_lora_model',True)
        self.do_train=kwargs.get('do_train',True)

        self.test_mode=kwargs.get('test_mode', False)
    
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
            # Use the provided 'output_dir' if available; otherwise, fall back to '_path' (labhandler) attribute if it exists.
            output_dir=kwargs.get('output_dir', getattr(self, '_path', '.')),
            auto_find_batch_size=kwargs.get('auto_find_batch_size',True),
            per_device_train_batch_size=kwargs.get('per_device_train_batch_size',8),
            gradient_accumulation_steps=kwargs.get('gradient_accumulation_steps',1),
            learning_rate=kwargs.get('learning_rate',0.0002),
            optim=kwargs.get('optim','paged_adamw_8bit'),
            save_strategy=kwargs.get('save_strategy','no'),
            save_total_limit=kwargs.get('save_total_limit',1),
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

            data_df=self.labh.handle_parameter(locals(),'data_df')
            llm_pipe=self.labh.handle_parameter(locals(),'llm_pipe')
            evaluators=self.labh.handle_parameter(locals(),'evaluators')

            # data_df=self.labh.handle_object(locals(),'data_df')
            # llm_pipe=self.labh.handle_object(locals(),'llm_pipe')
            # evaluators=self.labh.handle_object(locals(),'evaluators')
            self.sft_config['output_dir']=getattr(self,'_path', self.sft_config['output_dir'])


        if isinstance(data_df, pd.DataFrame):
            self.data_df = data_df.copy(); del data_df
        
        if _isinstance(llm_pipe, LLMPipeline):
            llm_pipe = self.fit_llm_pipe(llm_pipe)
            self.llm_pipe = llm_pipe

        if isinstance(evaluators, list) and all([_isinstance(e, LLMEvaluator) for e in evaluators]):
            self.evaluators=evaluators
        
        if getattr(self, 'test_mode', False):
            self.sft_config['num_train_epochs']=2
      
    def load_data(self, **kwargs) -> None:
        self.logger.info(f"Loading data")

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

        if isinstance(model, PeftModel) or (hasattr(model,'peft_config')):
            self.logger.info('Model is already prepared for lora')
            return model


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
        evaluators=kwargs.get('evaluators', self.evaluators)

        for evaluator in evaluators:
            evaluator_callback=callback_class(
                experiment=self,
                evaluator=evaluator, 
                **kwargs)
            evaluator_callback.on_epoch_end(state=TrainerState(epoch=0), model=self.llm_pipe.get_model(), **kwargs)

        return
    
    def train(self, **kwargs) -> None:

        self.logger.info(f"Start Training")

        with open(f"{self.sft_config['output_dir']}/trainer_args.yaml", 'w') as file:
            yaml.dump(self.trainer.args.to_dict(), file, sort_keys=False, default_flow_style=False)
        
        self.timestamp_train_start=get_timestamp()
        self.trainer.train()
        
        self.timestamp_train_done=get_timestamp()

        if hasattr(self, 'save_config'): self.save_config()
        return

    def run_entry(self, **kwargs) -> None:
        self.timestamp_start=get_timestamp()
        self.logger.info(f"Run Experiment")
        self.llm_pipe.prepare_inference()
        self.load_data(**kwargs)
        return

    def run(self, **kwargs) -> None:
        self.run_entry(**kwargs)

        if self.do_eval_base_model:
            self.evaluate_base_model(**kwargs)
        
        if (self.do_eval_lora_model or self.do_train):
            self.load_trainer(**kwargs)

        if self.do_eval_lora_model:
            for evaluator in self.evaluators:
                evaluator_callback=EvaluatorCallback(self, evaluator, **kwargs)
                self.trainer.add_callback(evaluator_callback)
        
        if self.do_train:
            for trainer_callback in [MCECallback(self), HistoryCallback(self)]:
                self.trainer.add_callback(trainer_callback)
            self.train(**kwargs)
        
        self.run_exit(**kwargs)
        return

    def run_exit(self, **kwargs) -> None:
        self.done=True
        self.timestamp_done=get_timestamp()
        self.logger.info(f"Done Experiment")
        if hasattr(self, 'save_config'): self.save_config()


    

