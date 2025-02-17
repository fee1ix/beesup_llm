from beesup_llm import *
from beesup_llm.toolkit.setup_utils import *
from beesup_llm.toolkit.llm_utils import *

from beesup_llm.dataset import BaseDataset
from beesup_llm.model_pipelines import *
import logging

import math
import torch
import types



class FinetuningPipeline(BaseDirectory):
    type='finetuning_pipeline'

    @staticmethod
    def matches(ref):
        if getattr_or_key(ref, 'type') == ['training']: return True
        if SFTLoraFinetuningPipeline.matches(ref): return True
        return False

    @classmethod
    def from_ref(cls, ref=None, **kwargs):
        kwargs.update(get_cls_attrs(cls))
        cls.logger.debug(f"{cls} ref={ref}, kwargs = {kwargs}\n")

        pre_config = get_config_from_ref(ref, **kwargs)
    
        if SFTLoraFinetuningPipeline.matches(pre_config):
            return SFTLoraFinetuningPipeline(ref=pre_config, **kwargs)
        
        return cls(ref=pre_config, **kwargs)

    def __init__(self, ref=None, llm_ref=None, dataset_ref=None, **kwargs):
        super().__init__(ref, **kwargs)

        self._default_config=dict(
            trainer_config=dict(
                seed=33,
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
                remove_unused_columns=False,
                report_to="none",
            ),
            data_collator_config=dict(
                padding='longest',
                label_pad_token_id =-100,
            )
        )


        self._config_key_order.extend([k for k in self._default_config.keys() if k not in self._config_key_order])
        self._config_keys_to_exclude.extend(['modelwrap','model','dataset','trainer','model_config','llm_ref','dataset_ref','llm_pipe'])

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
        
    def __call__(self, **kwargs):
        pass


import pickle
from trl import SFTTrainer, SFTConfig
from transformers.trainer import *
from transformers import TrainingArguments, DataCollatorForSeq2Seq

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
        # with open(f"{epoch}-{global_step}-inputs.pkl", "wb") as f: pickle.dump(inputs, f)
        # with open(f"{epoch}-{global_step}-loss.pkl", "wb") as f: pickle.dump(loss, f)
        # with open(f"{epoch}-{global_step}-outputs.pkl", "wb") as f: pickle.dump(outputs, f)
        

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


from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
class SFTLoraFinetuningPipeline(FinetuningPipeline):

    @staticmethod
    def matches(ref):
        if hasattrs_or_keys(ref,['sft_config','lora_config']): return True
        return False

    def __init__(self, ref=None, **kwargs):
        super().__init__(ref, **kwargs)

        self._default_config=dict(
            lora_config=dict(
                r=32,
                lora_alpha=3,
                use_rslora=True,
                target_modules='all-linear',
                lora_dropout=0.05,
                bias='none',
                task_type='CAUSAL_L',
            ),
            sft_config=dict(
                max_seq_length=4096,
                packing=False
            ),
        )
        self._config_key_order.extend([k for k in self._default_config.keys() if k not in self._config_key_order])

        self.update_config(self._default_config, overwrite_if_conflict=False)
        self.update_config_smart(kwargs)


    def get_lora_model(self, model, **kwargs):
        
        if isinstance(model, PeftModel) or (hasattr_or_key(model,'peft_config')):
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
    

    def load_trainer(self, model=None, **kwargs):

        if model is None: model = self.llm_pipe.get_model()

        lora_model = self.get_lora_model(model, **kwargs)
        lora_model.config.use_cache = False

        lora_config=LoraConfig(**self.get_updated_config(kwargs, config_key='lora_config'))

        trainer_config=self.get_updated_config(kwargs, config_key='trainer_config')
        #trainer_config=TrainingArguments(**trainer_config)
        sft_config=self.get_updated_config(kwargs, config_key='sft_config')
        #sft_config=SFTConfig(**sft_config)
        trainer_config=SFTConfig(**trainer_config,**sft_config)

        tokenizer = kwargs.get('tokenizer', self.llm_pipe.get_training_tokenizer())
        data_collator_config=self.get_updated_config(kwargs, config_key='data_collator_config')
        data_collator=CustomDataCollatorForSeq2Seq(tokenizer=tokenizer,model=lora_model,**data_collator_config)

        
        self.trainer = CustomSFTTrainer(
            model=lora_model,
            train_dataset=kwargs.get('train_dataset', getattr(self, 'train_dataset', None)),
            eval_dataset=kwargs.get('eval_dataset', getattr(self, 'eval_dataset', None)),
            peft_config=lora_config,
            args=trainer_config,  
            data_collator=data_collator,
            #**sft_config
        )







    


