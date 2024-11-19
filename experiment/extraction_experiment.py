from beesup_llm import *
from ..toolkit.setup_utils import *
from ..toolkit.llm_utils import *

from beesup_llm.model import *
from beesup_llm.dataset import BaseDataset
from beesup_llm.training import *
from beesup_llm.extraction.extraction_pipeline import *



class ExtractionExperiment(BaseDirectory):
    type='extraction_experiment'

    def __init__(self, ref=None, dataset_ref=None, pipeline_ref=None, model_ref=None, trainer_ref=None, **kwargs):
        super().__init__(ref)

        self._config_key_order.extend([])
        self._config_keys_to_exclude.extend(['dataset_df','train_df','test_df','eval_df','train_ds','eval_ds','trainer','modelwrap','pipeline','dataset'])

        self._default_config=dict(
            done = False,
            batch_size = 4,

            evaluate_base_model=True,
            generation_config=dict(
                do_sample=False,
            ),

            trainer_args=dict(
                seed=42,
                num_train_epochs=2,
                learning_rate=0.0002,
                output_dir=f"{self.path}",
                save_strategy='no',
                logging_strategy='steps',
                logging_steps=1,
                logging_first_step=True,
                do_eval=False,
            ),

        )

        self.update_attributes(self._default_config, overwrite=False)

        if model_ref is not None:
            self.modelwrap=GenModelWrap.from_ref(model_ref)

        if dataset_ref is not None:
            self.dataset=BaseDataset.from_ref(dataset_ref)

        if trainer_ref is not None:
            self.trainwrap=BaseTrainerWrap.from_ref(trainer_ref)

        if pipeline_ref is not None:
            self.pipeline=ExtractionPipeline.from_ref(pipeline_ref)

    def load_data(self, **kwargs):
        self.logger.info(f"Loading Data")

        assert hasattr(self, 'dataset'), "Dataset must be assigned before loading data"
        assert hasattr(self, 'pipeline'), "Extraction Pipeline must be assigned before loading data"
        assert hasattr(self, 'modelwrap'), "ModelWrap must be assigned before loading data"

        train_df=self.dataset.get_df_splits('train')
        eval_df=self.dataset.get_df_splits(['test','eval'])

        self.modelwrap.load_training_tokenizer()
        self.modelwrap.load_inference_tokenizer()

        train_df=self.pipeline.prepare_df_for_finetuning(train_df)
        train_ds=self.pipeline.get_ds_for_finetuning(train_df, self.modelwrap.get_training_tokenizer()) # tokenizer type doesnt matter at this point, because padding is not applied

        eval_df=self.pipeline.prepare_df_for_completion(eval_df)
        eval_ds=self.pipeline.get_ds_for_completion(eval_df,  self.modelwrap.get_inference_tokenizer())

        self.train_df=train_df
        self.eval_df=eval_df

        self.train_ds=train_ds
        self.eval_ds=eval_ds

    def load_trainer(self,**kwargs):
        self.logger.info(f"Loading Trainer")
        
        self.load_data(**kwargs)

        trainer_args=self.get_updated_config(kwargs, 'trainer_args')
        model=self.modelwrap.get_model()

        trainer= self.trainwrap.get_trainer(
            model=model,
            tokenizer=self.modelwrap.get_training_tokenizer(),
            train_dataset=self.train_ds,
            args=trainer_args,
        )
        self.trainer=trainer

    def get_trainer(self, **kwargs):

        trainer=getattr(self, 'trainer', None)
        if trainer is None:
            self.load_trainer()
            trainer=self.trainer
            del self.trainer
            return trainer
        
        else:
            return self.trainer




    def run(self,**kwargs):

        self.trainwrap.run(**kwargs)




