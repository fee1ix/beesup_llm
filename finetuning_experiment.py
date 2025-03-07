
from beesup_llm import get_labhandler, _isinstance
from beesup_llm.llm import LLMPipeline

import logging
import pandas as pd

import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback, TrainingArguments, DataCollatorForSeq2Seq
from transformers.trainer import TrainerState
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training

class LoRAExperiment(object):
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
            llm_pipe=None,
            train_df=None,
            evaluators:list=[],
            labh=get_labhandler(),
            **kwargs):
        
        self.label=label
        self.seed=kwargs.get('seed',42)
        
        self.do_eval_base_model=kwargs.get('do_eval_base_model',True)
        self.do_eval_lora_model=kwargs.get('do_eval_lora_model',True)
        self.do_finetune=kwargs.get('do_finetune',True)
    
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
            per_device_train_batch_size=kwargs.get('per_device_train_batch_size',4),
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
            packing=kwargs.get('packing',False)
        ))


        if labh is not None:
            self.labh=labh
            self.labh.attach_parent(locals())
            train_df=self.labh.handle_object(locals(),'train_df')
            llm_pipe=self.labh.handle_object(locals(),'llm_pipe')
            evaluators=self.labh.handle_object(locals(),'evaluators')


        if isinstance(train_df, pd.DataFrame):
            self.train_df = train_df.copy(); del train_df
        
        if _isinstance(llm_pipe, LLMPipeline):
            llm_pipe = self.fit_llm_pipe(llm_pipe)
            self.llm_pipe = llm_pipe

        if evaluators:
            self.evaluators=evaluators
    

    # def load_data(self, **kwargs):

    #     assert hasattr(self, 'train_df'), "train_df is missing"
    #     assert hasattr(self, 'llm_pipe'), "llm_pipe is missing"





    def get_lora_model(self, model: torch.nn.Module) -> PeftModel:
        self.logger.info(f"Loading Lora model")

        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, LoraConfig(**self.lora_config))
        model.print_trainable_parameters()

        return model
    

    def load_trainer(self, lora_model: PeftModel) -> None:
        self.logger.info(f"Loading trainer")

        lora_model.config.use_cache = False

        data_collator=DataCollatorForSeq2Seq(
            tokenizer=self.llm_pipe.get_tokenizer('training'),
            model=lora_model,
            padding='longest',
            label_pad_token_id =-100
            )
        
        self.trainer=SFTTrainer(
            model=lora_model,
            data_collator=data_collator,
            train_dataset=self.train_ds,
            peft_config=self.lora_config,
            args=self.sft_config,
            )
        
        return   


    
   


    

