from beesup_llm import *
from ..toolkit.setup_utils import *
from ..toolkit.llm_utils import *

from beesup_llm.dataset import BaseDataset
from beesup_llm.model import BaseModelWrap
import logging


import torch

import types

from trl import SFTTrainer
from transformers.trainer import *
from transformers import TrainingArguments
from transformers import DataCollatorForSeq2Seq


class BaseTraining(BaseDirectory):

    def __new__(cls, ref=None, dataset_ref=None, model_ref=None):

        instance = super().__new__(cls)
        instance.__init__(ref, dataset_ref, model_ref)

        if hasattr(instance, 'lora_config'):
 
            instance = super(BaseTraining,LoraTraining).__new__(LoraTraining)
        
        else:
            instance = super().__new__(cls)
  
        return instance

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
        self.dataset_config=self._dataset.get_config()

        self._modelwrap=BaseModelWrap(model_ref)
        self.model_config=self._modelwrap.get_config()


        self._default_config=dict(
            done = False,
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
        # if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
        #     self.optimizer.eval()

        # Optional logging for batch size and dataset size
        batch_size = args.eval_batch_size
        # self.logger.info(f"\n***** Running {description} *****")
        # self.logger.info(f"  Batch size = {batch_size}")
        # if hasattr(dataloader, "dataset"):
        #     self.logger.info(f"  Num examples = {len(dataloader.dataset)}")

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
        


        # # Initialize metrics and perform the evaluation loop
        # all_preds, all_labels, all_losses = [], [], []
        # for step, inputs in enumerate(dataloader):
        #     with torch.no_grad():
        #         losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only=False)
        #         all_losses.append(losses)
        #         all_preds.append(logits)
        #         all_labels.append(labels)

        # # Gather metrics, averaging loss if available
        # avg_loss = torch.stack(all_losses).mean().item() if all_losses else None
        # metrics = {f"{metric_key_prefix}_loss": avg_loss} if avg_loss is not None else {}

        # return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=len(dataloader.dataset))


class LoraTraining(BaseTraining):

    def __init__(self, ref=None, dataset_ref=None, model_ref=None):
        super().__init__(ref, dataset_ref, model_ref)

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

    
    def get_lora_model(self, model):

        self.logger.info(f"{self.name.upper()}\tprepare lora model")

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

        set_config(self.get_config())

        return model
    
    def get_trainer(self, **kwargs):

        model=kwargs.get('model', None)
        if model is None:
            model = self._modelwrap.get_model()
            model = self.get_lora_model(model)


        tokenizer = self._modelwrap.get_training_tokenizer()

        train_ds, eval_ds, test_ds = self._dataset.arrange(tokenizer)

        from peft import LoraConfig

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            peft_config=LoraConfig(**self.lora_config),
            args=TrainingArguments(**self.args),  
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer,model=model,**self.data_collator_config),
            **self.sftt_args
        )

        trainer.evaluation_loop=types.MethodType(self.custom_evaluation_loop,trainer)

        return trainer
    





    


    
    





