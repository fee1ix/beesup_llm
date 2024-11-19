from beesup_llm import *
from ..toolkit.setup_utils import *
from ..toolkit.llm_utils import *

from beesup_llm.dataset import BaseDataset
from beesup_llm.model import *
import logging


import torch
import types

from trl import SFTTrainer
from transformers.trainer import *
from transformers import TrainingArguments
from transformers import DataCollatorForSeq2Seq


class BaseTrainerWrap(BaseDirectory):
    type = 'trainer'

    @staticmethod
    def matches(ref):
        if getattr_or_key(ref, 'type') == ['training']: return True
        if LoraTraining.matches(ref): return True
        return False

    @classmethod
    def from_ref(cls, ref=None, **kwargs):
        kwargs.update(get_cls_attrs(cls))
        cls.logger.debug(f"{cls} ref={ref}, kwargs = {kwargs}\n")

        pre_config = get_config_from_ref(ref, **kwargs)
    
        if LoraTrainerWrap.matches(pre_config):
            return LoraTrainerWrap(ref=pre_config, **kwargs)
        
        return cls(ref=pre_config, **kwargs)

    def __init__(self, ref=None, model_ref=None, dataset_ref=None, **kwargs):
        super().__init__(ref,  model_ref=None, dataset_ref=None, **kwargs)

        self._config_key_order.extend([])
        self._config_keys_to_exclude.extend(['modelwrap','model','dataset','trainer','model_config'])
        self._model_is_prepared=False

        self._default_config=dict(
            args=dict(
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
            sftt_args=dict(
                max_seq_length=4096,
                packing=False
            ),
            
            data_collator_config=dict(
                padding='longest',
                label_pad_token_id =-100,
            )
        )
        self._config_key_order.extend([k for k in self._default_config.keys() if k not in self._config_key_order])
        self.update_attributes(self._default_config, overwrite=False)

        if model_ref is not None:
            self.modelwrap=GenModelWrap.from_ref(model_ref)

        if dataset_ref is not None:
            self.dataset=BaseDataset.from_ref(dataset_ref)

    def get_model(self):
        return getattr(self,'model',self.modelwrap.get_model())

    def get_tokenizer(self):

        if hasattr(self,'modelwrap'): 
            return self.modelwrap.get_training_tokenizer()


     
    def run(self, trainer, **kwargs):

        #trainer = kwargs.get('trainer', self.get_trainer(**kwargs))

        self.logger.info(f"Running {self.name.upper()}")

        trainer.train()

        self.logger.info(f"Completed")

        self.done=True
        self.datetime_end=self.get_datetime()
        set_config(self.get_config())

        import gc
        gc.collect()
        torch.cuda.empty_cache()
    
    @staticmethod
    def custom_evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Minimal prediction/evaluation loop to prepare the model in evaluation state
        and maintain compatibility with the original training setup.
        """
        args = self.args
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        # Handle model preparation if needed
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = self.accelerator.prepare_model(model, evaluation_mode=True)
            self.model = model if self.is_fsdp_enabled else self.model
            if model is not self.model:
                self.model_wrapped = model
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # Set model to evaluation mode
        model.eval()

        batch_size = args.eval_batch_size


        print(vars(dataloader))

        modelwrap=BaseModelWrap(model)
        modelwrap.load_inference_tokenizer()

        inference_outputs=modelwrap.inference_loop(dataloader)
        inference_df=get_inference_df(inference_outputs)

        save_path=f'{self.args.output_dir}/checkpoint-{self.state.global_step}-inference_df.pkl'
        inference_df.to_pickle(save_path)

        # Example dimensions - adjust as per your dataloader and model output shapes
        batch_size = self.args.eval_batch_size
        num_batches = len(dataloader)
        num_classes = 10  # Example, adjust as per model output

        # Dummy placeholders shaped like real outputs
        all_preds = [torch.zeros(batch_size, num_classes) for _ in range(num_batches)]
        all_labels = [torch.zeros(batch_size) for _ in range(num_batches)]
        all_losses = [torch.tensor(0.0) for _ in range(num_batches)]

        # Gather metrics, with average loss as a dummy value
        avg_loss = torch.stack(all_losses).mean().item()  # Example average loss calculation
        metrics = {f"{metric_key_prefix}_loss": avg_loss}

        # Convert lists of tensors into a single tensor or numpy array if required by EvalLoopOutput
        all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
        all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
        all_losses = torch.stack(all_losses).cpu().numpy()

        # Wrap in EvalLoopOutput for compatibility
        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=len(all_labels))


from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training

class LoraTrainerWrap(BaseTrainerWrap):
    type = 'lora_trainer'

    @staticmethod
    def matches(ref):
        if hasattr_or_key(ref,'lora_config'): return True
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
        )
        self._config_key_order.extend([k for k in self._default_config.keys() if k not in self._config_key_order])

        self.update_attributes(self._default_config, overwrite=False)

    def prepare_model_for_lora(self, model, **kwargs):

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
    
    def get_trainer(self, **kwargs):

        # model = kwargs.get('model', self.get_model())

        lora_model = self.prepare_model_for_lora(**kwargs)
        lora_model.config.use_cache = False

        tokenizer = kwargs.get('tokenizer', self.get_tokenizer())
        
        lora_config=LoraConfig(**self.get_updated_config(kwargs, config_key='lora_config'))
        args=TrainingArguments(**self.get_updated_config(kwargs, config_key='args'))

        data_collator_config=self.get_updated_config(kwargs, config_key='data_collator_config')
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer,model=lora_model,**data_collator_config)
        sftt_args=self.get_updated_config(kwargs, config_key='sftt_args')

        trainer = SFTTrainer(
            model=lora_model,
            train_dataset=kwargs.get('train_dataset', getattr(self, 'train_dataset', None)),
            eval_dataset=kwargs.get('eval_dataset', getattr(self, 'eval_dataset', None)),
            peft_config=lora_config,
            args=args,  
            data_collator=data_collator,
            **sftt_args
        )

        return trainer


    def run(self, **kwargs):

        self.trainer.train()
        



        


    # def get_trainer(self, **kwargs):

    #     model=kwargs.get('model', None)
    #     if model is None:
    #         model = self._modelwrap.get_model()
    #         model = self.get_lora_model(model)


    #     tokenizer = self._modelwrap.get_training_tokenizer()

    #     train_ds, eval_ds, test_ds = self._dataset.arrange(tokenizer)

        

    #     trainer = SFTTrainer(
    #         model=model,
    #         train_dataset=train_ds,
    #         eval_dataset=eval_ds,
    #         peft_config=LoraConfig(**self.lora_config),
    #         args=TrainingArguments(**self.args),  
    #         data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer,model=model,**self.data_collator_config),
    #         **self.sftt_args
    #     )

    #     trainer.evaluation_loop=types.MethodType(self.custom_evaluation_loop,trainer)

    #     return trainer
    





    


    
    





