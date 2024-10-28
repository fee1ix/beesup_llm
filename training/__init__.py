from beesup_llm import *
from ..toolkit.setup_utils import *

from beesup_llm.dataset import BaseDataset
from beesup_llm.model import BaseModelWrap
import logging


class BaseTraining(BaseDirectory):

    # def __new__(cls, ref=None):

    #     temp_instance = super().__new__(cls)
    #     temp_instance.__init__(ref)

    #     if hasattr(temp_instance, 'name_or_path'):
    #         if temp_instance.name_or_path == 'meta-llama/Meta-Llama-3.1-8B-Instruct':
    #             return super(BaseModelWrap,LlamaModelWrap).__new__(LlamaModelWrap)
        
    #     return super().__new__(cls)


    def __init__(self, ref=None, dataset_ref=None, model_ref=None):

        self.type='training'
        super().__init__(ref)

        self._config_key_order.extend([])
        self._config_keys_to_exclude.extend(['model','inference_tokenizer','training_tokenizer'])

        if hasattr(self, 'dataset_config'):
            dataset_ref = self.dataset_config
        
        if hasattr(self, 'model_config'):
            model_ref = self.model_config

        self._dataset=BaseDataset(dataset_ref)
        self.dataset_df=self.prepare_dataset_df(self._dataset.dataset_df)
        self.dataset_config=self._dataset.get_config()

        self._modelwrap=BaseModelWrap(model_ref)
        self.model_config=self._modelwrap.get_config()


        self._default_config=dict(
            done = False,
            eval_batch_size = 4,
            use_dataset_splits=['train','eval'],

            lora_config=dict(
                r=32,
                lora_alpha=3,
                use_rslora=True,
                target_modules='all-linear',
                lora_dropout=0.05,
                bias='none',
                task_type='CAUSAL_L',
            ),

            training_args=dict(
                seed=42,
                auto_find_batch_size=True,
                gradient_accumulation_steps=1,
                gradient_checkpointing_kwargs=dict(
                    use_reentrant=False,
                ),
                warmup_steps=0,
                num_train_epochs=12,
                learning_rate=0.0002,
                output_dir=f"{self.path}",
                optim='paged_adamw_8bit',
                per_device_eval_batch_size=16,
                save_strategy='no',
                logging_strategy='steps',
                logging_steps=1,
                logging_first_step=True,
                do_eval=True,
                eval_strategy='epoch',
                prediction_loss_only=False,
            ),
            training_config=dict(
                max_seq_length=4096,
                packing=False
            ),
        )

        self.update_attributes(self._default_config, overwrite=False)

    

class LoraTraining(BaseTraining):

    def __init__(self, ref=None, dataset_ref=None, model_ref=None):
        super().__init__(ref, dataset_ref, model_ref):

    def get_prepared_lora_model(self):

        self.logger.info(f"{self.name.upper()}\tPREPARE LORA-MODEL...")

        if self.done:
            self.logger.warning(f"{self.name.upper()} already completed")
            return

        if not hasattr(self._modelwrap,'model'):
            self._modelwrap.load_model()

        if not hasattr(self._modelwrap,'training_tokenizer'):
            self._modelwrap.load_training_tokenizer()

        model=self._modelwrap.model
        tokenizer=self._modelwrap.training_tokenizer


        from peft import prepare_model_for_kbit_training
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(**self.lora_config)

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        #SET LORA INFO
        n_trainable_params,n_total_params=model.get_nb_trainable_parameters()
        p_trainable_params=n_trainable_params/n_total_params

        import math
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
    
    





