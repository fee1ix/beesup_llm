from beesup_llm import *
from beesup_llm.toolkit.setup_utils import *
from beesup_llm.toolkit.llm_utils import *

from beesup_llm.dataset import BaseDataset
from beesup_llm.model_pipelines import *
import logging

import math
import torch
import types

from trl import SFTTrainer
from transformers.trainer import *
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training



class SFTLoraTrainer(BaseDirectory):
    type = 'trainer'

    def __init__(self, ref=None, llm_ref=None, dataset_ref=None, **kwargs):
        super().__init__(ref, **kwargs)

        self._model_is_prepared=False
        self._default_config=dict(
            trainer_args=dict(
                seed=33,
                #auto_find_batch_size=True,
                auto_find_batch_size=False,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=1,
                gradient_checkpointing_kwargs=dict(
                    use_reentrant=False,
                ),
                warmup_steps=0,
                num_train_epochs=12,
                learning_rate=0.0002,
                output_dir=f"{self._path}",
                optim='paged_adamw_8bit',
                per_device_eval_batch_size=16,
                save_strategy='no',
                logging_strategy='steps',
                logging_steps=1,
                logging_first_step=True,
                do_train=True,
                do_eval=True,
                eval_strategy='epoch',
                prediction_loss_only=False,
            ),
            data_collator_config=dict(
                padding='longest',
                label_pad_token_id =-100,
            ),
            lora_config=dict(
                r=32,
                lora_alpha=3,
                use_rslora=True,
                target_modules='all-linear',
                lora_dropout=0.05,
                bias='none',
                task_type='CAUSAL_L',
            ),
            sft_trainer_args=dict(
                max_seq_length=4096,
                packing=False
            ),
        )


        self._config_key_order.extend([k for k in self._default_config.keys() if k not in self._config_key_order])
        self._config_keys_to_exclude.extend(['model','dataset','trainer','model_config','llm_ref','dataset_ref'])

        self.update_config(self._default_config, overwrite_if_conflict=False)
        self.update_config_smart(
            kwargs, 
            interpret_none_as_val=True, 
            overwrite_if_conflict=True, 
            allow_new_atomic_keys=False, 
            allow_new_nested_keys=False
        )

        if llm_ref:
            self.llm_pipe=LanguageModelPipeline.from_ref(llm_ref)

        if dataset_ref:
            self.dataset=BaseDataset.from_ref(dataset_ref)

    def get_prepared_model(self, model, **kwargs):

        if isinstance(model, PeftModel):
            self.logger.info('Model already prepared for lora')
            return model
    
        #model=model.unload()
        self.logger.info('Preparing lora model')
        
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        lora_config=self.get_updated_config(kwargs, config_key='lora_config')
        lora_config = LoraConfig(**lora_config)

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        #SET LORA INFO
        n_trainable_params,n_total_params=model.get_nb_trainable_parameters()
        p_trainable_params=n_trainable_params/n_total_params

        
        if self.lora_config.get('use_rslora',False):
            lora_scale=self.lora_config['lora_alpha']/math.sqrt(self.lora_config['r'])
        else:
            lora_scale=self.lora_config['lora_alpha']/self.lora_config['r']

        self.lora_info={
            'n_trainable_params':n_trainable_params,
            'n_total_params':n_total_params,
            'p_trainable_params':p_trainable_params,
            'lora_scale':lora_scale,
        }

        return model
    
    def load(self, **kwargs):
        lora_model = self.get_prepared_model(**kwargs)
        lora_model.config.use_cache = False

        tokenizer = kwargs.get('tokenizer', self.llm_pipe.get_training_tokenizer())
        
        lora_config=LoraConfig(**self.get_updated_config(kwargs, config_key='lora_config'))
        trainer_args=TrainingArguments(**self.get_updated_config(kwargs, config_key='trainer_args'))
        data_collator_config=self.get_updated_config(kwargs, config_key='data_collator_config')

        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer,model=lora_model,**data_collator_config)
        sft_trainer_args=self.get_updated_config(kwargs, config_key='sft_trainer_args')

        self.trainer = SFTTrainer(
            model=lora_model,
            train_dataset=kwargs.get('train_dataset', getattr(self, 'train_dataset', None)),
            eval_dataset=kwargs.get('eval_dataset', getattr(self, 'eval_dataset', None)),
            peft_config=lora_config,
            args=trainer_args,  
            data_collator=data_collator,
            **sft_trainer_args
        )
    
    def get_trainer(self, **kwargs):

        if not hasattr(self, 'trainer'):
            self.load(**kwargs)
        
        return self.trainer
    

    


