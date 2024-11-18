from beesup_llm import *
from ..toolkit.setup_utils import *
from ..toolkit.llm_utils import *

from beesup_llm.model import *
from beesup_llm.dataset import BaseDataset
from beesup_llm.training import *
from beesup_llm.extraction.extraction_pipeline import *

# from transformers import TrainerCallback
# class CustomEvalCallback(TrainerCallback):
#     def __init__(self, pipeline):
#         self.pipeline = pipeline
#         pass

#     def on_epoch_end(self, args, state, control, **kwargs):

#         trainer = kwargs['trainer']

#         model = args['model']

#         model = trainer._wrap_model(trainer.model, training=False, dataloader=args['eval_dataloader'])

class ExtractionExperiment(BaseDirectory):
    type='extraction_experiment'

    def __init__(self, ref=None, dataset_ref=None, pipe_ref=None, model_ref=None, trainer_ref=None, **kwargs):
        super().__init__(ref)

        self._config_key_order.extend([])
        self._config_keys_to_exclude.extend(['dataset_df','train_df','test_df','eval_df'])

        self._default_config=dict(
            done = False,
            batch_size = 4,

            evaluate_base_model=True,
            generation_config=dict(
                do_sample=False,
            ),
        )

        self.update_attributes(self._default_config, overwrite=False)

        if model_ref is not None:
            self.modelwrap=GenModelWrap.from_ref(model_ref)

        if dataset_ref is not None:
            self.dataset=BaseDataset.from_ref(dataset_ref)
            self.train_df=self.dataset.get_df_split('train')
            self.eval_df=self.dataset.get_df_split('eval')
        
        if trainer_ref is not None:
            self.trainwrap=BaseTrainerWrap.from_ref(trainer_ref)

        if pipe_ref is not None:
            self.pipe=ExtractionPipeline.from_ref(pipe_ref)


            
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


    def prepare(self,**kwargs):
        
        dataset_df=self.dataset.dataset_df
        dataset_df=self.pipe.prepare_df(dataset_df, **kwargs)

        from datasets import Dataset
        train_df=dataset_df[dataset_df.split=='train']
        train_ds=Dataset.from_list(train_df.apply(lambda x: prepare_sample_for_chat_completion(x, self.modelwrap.get_inference_tokenizer()),axis=1).to_list())

        test_df=dataset_df[dataset_df.split=='test']
        test_ds=Dataset.from_list(test_df.apply(lambda x: prepare_sample_for_chat_completion(x, self.modelwrap.get_inference_tokenizer()),axis=1).to_list())

        self.trainwrap.model=self.modelwrap.model

        self.trainwrap.prepare(
            train_dataset=train_ds,
            **kwargs
        )

        self.train_df=train_df
        self.train_ds=train_ds

        self.test_df=test_df
        self.test_ds=test_ds
    
    def run(self,**kwargs):

        self.trainwrap.run(**kwargs)




