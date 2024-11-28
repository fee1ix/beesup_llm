import beesup_llm
from beesup_llm import *
from beesup_llm.toolkit.setup_utils import *
from beesup_llm.toolkit.llm_utils import *
from beesup_llm.toolkit.system import *

from beesup_llm.model import *
from beesup_llm.dataset import BaseDataset
from beesup_llm.training import *
from beesup_llm.extraction.extraction_pipeline import *


from transformers import TrainerCallback
class EvaluationCallback(TrainerCallback):
    def __init__(self, experiment):
        self.experiment = experiment
        pass

    def get_eval_df(self, model):
        eval_df=self.experiment.eval_df
        eval_df=self.experiment.pipeline(eval_df, model_ref=model)
        return eval_df
    
    def save_eval_df(self, eval_df, global_step):
        eval_df.to_pickle(f"{self.experiment.path}/{str(global_step).zfill(4)}_eval_df.pkl")
        self.experiment.logger.info(f"Saved eval_df to {self.experiment.path}/{str(global_step).zfill(4)}_eval_df.pkl")

    def on_epoch_end(self, args, state, control, **kwargs):
        model=kwargs['model']
        model.eval()
        eval_df=self.get_eval_df(model)
        self.save_eval_df(eval_df,state.global_step)
        

class ExtractionExperiment(BaseDirectory):
    type='extraction_experiment'

    @classmethod
    def spawn_multirun_config(cls, the_input=None):
 
        if isinstance(the_input, pd.DataFrame):
            multirun_df=the_input

        elif isinstance(the_input, list):
            if all(isinstance(x, int) for x in the_input):
                overview_df=cls.get_overview()
                multirun_df=overview_df[overview_df['id'].isin(the_input)]

        elif the_input is None:
            overview_df=cls.get_overview(keypaths=['done'])
            multirun_df=overview_df[overview_df['done']==False].copy()
            multirun_df.reset_index(drop=True, inplace=True)
        

        multirun_config=dict(
            framework_dirs=[os.path.dirname(path) for path in beesup_llm.__path__], #add as sys path in the run script
            module_path=__file__,
            script_path=f"{os.path.dirname(__file__)}/multirun_script.py",
            experiment_dirs=multirun_df.path.values.tolist(),
        )

        save_yaml(multirun_config, f"{cls.get_dir_path()}/multirun_config.yaml")
        cls.logger.info(f"Saved multirun_config to {cls.get_dir_path()}/multirun_config.yaml")

        print()
        print("Run the following command to start the multirun:")
        print(f"\tconda activate beesup; python {multirun_config['script_path']} {cls.get_dir_path()}/multirun_config.yaml")
        print()

        return multirun_df


    def __init__(self, ref=None, dataset_ref=None, pipeline_ref=None, model_ref=None, trainer_ref=None, **kwargs):
        super().__init__(ref, **kwargs)

        self._config_key_order.extend([])
        self._config_keys_to_exclude.extend(['dataset_df','train_df','test_df','eval_df','train_ds','eval_ds','trainer','trainwrap','modelwrap','pipeline','dataset'])

        self._default_config=dict(
            done = False,
            seed = 55,
            do_eval_base_model=True,

            generation_config=dict(
                do_sample=False,
            ),

            dataloader_config=dict(
                batch_size=8,
            ),

            trainer_args=dict(
                num_train_epochs=10,
                learning_rate=0.0002,
                output_dir=f"{self.path}",
                save_strategy='no',
                logging_strategy='steps',
                logging_steps=1,
                logging_first_step=True,
                eval_strategy='no',
                do_eval=False,
            ),

        )

        self.update_config(self._default_config, overwrite_if_conflict=False)
        self.trainer_args['seed']=self.seed

        model_ref = model_ref or getattr(self, 'model_config', None)
        if model_ref is not None:
            self.modelwrap=GenModelWrap.from_ref(model_ref)
            if not self.modelwrap.is_spawned(): self.modelwrap.spawn()
            self.model_config=self.get_updated_sub_config(self.modelwrap.get_config())

        dataset_ref = dataset_ref or getattr(self, 'dataset_config', None)
        if dataset_ref is not None:
            self.dataset=BaseDataset.from_ref(dataset_ref)
            if not self.dataset.is_spawned(): self.dataset.spawn()
            self.dataset_config=self.get_updated_sub_config(self.dataset.get_config())

        trainer_ref = trainer_ref or getattr(self, 'trainer_config', None)
        if trainer_ref is not None:
            self.trainwrap=BaseTrainerWrap.from_ref(trainer_ref)
            if not self.trainwrap.is_spawned(): self.trainwrap.spawn()
            self.trainer_config=self.get_updated_sub_config(self.trainwrap.get_config())

        pipeline_ref = pipeline_ref or getattr(self, 'pipeline_config', None)
        if pipeline_ref is not None:
            self.pipeline=ExtractionPipeline.from_ref(pipeline_ref)
            if not self.pipeline.is_spawned(): self.pipeline.spawn()
            self.pipeline_config=self.get_updated_sub_config(self.pipeline.get_config())

    def get_updated_sub_config(self, sub_config):

            ignore_keys=['type', 'id', 'name', 'path', 'parent_dir_path', 'parent_lab_path','timestamp_init']

            self_config=self.get_config()
            self_config={k:v for k,v in self_config.items() if k not in ignore_keys}

            sub_keys=sub_config.keys()
            self_config={k:v for k,v in self_config.items() if k  in sub_keys}

            for key in sub_keys:
                if key in ignore_keys: continue
                if hasattr(self, key): self.__delattr__(key)

            updated_sub_config=update_nested_dict(sub_config,self_config, overwrite=True)

            return updated_sub_config

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

    def get_trainer(self, model=None, **kwargs):
        assert model is not None, "model must be passed"

        self.logger.info(f"Loading Trainer")

        # trainer_args=self.get_updated_config(kwargs, 'trainer_args')
        # self.logger.debug(f"trainer_args: {trainer_args}")

        trainer= self.trainwrap.get_trainer(
            model=model,
            tokenizer=self.modelwrap.get_training_tokenizer(),
            train_dataset=self.train_ds,
            #args=trainer_args,
        )

        trainer.add_callback(EvaluationCallback(self))

        return trainer
    
    def evaluate_base_model(self,model):

        eval_callback=EvaluationCallback(self)
        
        eval_df=eval_callback.get_eval_df(model)
        eval_callback.save_eval_df(eval_df,0)

    def run(self,**kwargs):

        self.logger.info(f"RUNNING")
        set_seeds(self.seed)

        if self.done:
            self.logger.info(f"Already done. Skipping.")
            return

        self.timestamp_run=get_timestamp()
        base_model=self.modelwrap.get_model()

        self.load_data(**kwargs)

        if self.do_eval_base_model:
            self.evaluate_base_model(model=base_model)

        trainer=self.get_trainer(model=base_model,**kwargs)
        trainer_args=trainer.args.to_dict()
        save_yaml(trainer_args, f"{self.path}/trainer_args.yaml")

        trainer.train()

        self.done=True
        self.timestamp_done=get_timestamp()
        set_config(self.get_config())
        return

    def get_evals_df(self):

        fns=os.listdir(self.path)
        fns=[fn for fn in fns if fn.endswith('eval_df.pkl')]
        evals_df=pd.DataFrame()
        for fn in fns:
            global_step=int(fn[:4])
            eval_df=pd.read_pickle(f"{self.path}/{fn}")
            eval_df['global_step']=global_step
            evals_df=pd.concat([evals_df,eval_df])

        evals_df.reset_index(drop=True, inplace=True)
        return evals_df




if __name__ == '__main__':
    import sys
    import logging

    if len(sys.argv) != 2:
        print("Usage: python experiment_module.py <config_path>")
        sys.exit(1)

    experiment_dir=sys.argv[1]
    os.chdir(experiment_dir)

    logging.basicConfig(
        level=logging.INFO,
        #level=logging.DEBUG,
        format='%(asctime)s - %(filename)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s')

    # Initialize and run the experiment
    experiment = ExtractionExperiment(experiment_dir)
    experiment.run()

    