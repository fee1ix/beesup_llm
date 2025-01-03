import beesup_llm
from beesup_llm import *
from beesup_llm.toolkit.setup_utils import *
from beesup_llm.toolkit.llm_utils import *
from beesup_llm.toolkit.system import *

#from beesup_llm.model import *
from beesup_llm.dataset import BaseDataset
from beesup_llm.training import *

from beesup_llm.model_pipelines import *
from beesup_llm.extraction.extraction_pipeline import *


from transformers import TrainerCallback

class PredictionCallback(TrainerCallback):
    def __init__(self, experiment):
        self.experiment = experiment
        self.name='pred_callback'
        pass

    def get_pred_df(self, model, eval_df=None):
        if eval_df is None: eval_df=self.experiment.eval_df

        pred_df=self.experiment.extractor_pipe(eval_df, llm_ref=model)
        #print(f"_recent_generation_config: {self.experiment.extractor_pipe.llm_pipe._recent_generation_config}")
        return pred_df

    def eval_loop(self, model, eval_batch_size):

        eval_df=self.experiment.eval_df

        pred_dfs=[]
        for i in range(0, len(eval_df),eval_batch_size):
            self.experiment.logger.info(f"Evaluate Sample {i+eval_batch_size}/{len(eval_df)}")
            pred_df=self.get_pred_df(model, eval_df.iloc[i:i+eval_batch_size].copy())
            pred_dfs.append(pred_df)
            
        pred_df=pd.concat(pred_dfs)

        return pred_df

    def save_df(self, df, global_step):
        save_path=f"{self.experiment._path}/{str(global_step).zfill(4)}_{self.name}_df.pkl"
        df.to_pickle(save_path)
        self.experiment.logger.info(f"Saved {self.name} to {save_path}")


    def on_epoch_end(self, args, state, control, **kwargs):
        self.experiment.logger.info(f"global step: {state.global_step}")
        model=kwargs['model']
        model.eval()

        pred_df=self.eval_loop(model, args.per_device_eval_batch_size)
        self.save_df(pred_df, state.global_step)

class ExtractionExperiment(BaseDirectory):
    type='extraction_experiment'

    @classmethod
    def spawn_multirun_config(cls, the_input=None):
 
        if isinstance(the_input, pd.DataFrame):
            multirun_df=the_input

        elif isinstance(the_input, list):
            if all(isinstance(x, int) for x in the_input):
                overview_df=cls.get_overview(keypaths=['path'])
                multirun_df=overview_df[overview_df['id'].isin(the_input)]

        elif the_input is None:
            overview_df=cls.get_overview(keypaths=['path','done'])
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
    
    def __init__(self, ref=None, dataset_ref=None, extractor_ref=None, llm_ref=None, trainer_ref=None, **kwargs): 
        super().__init__(ref, **kwargs)

        self._default_config=dict(
            done = False,
            seed = 55,
            do_eval_base_model=True,
            do_train=True,

            llm_config=dict(
                generation_config=dict(
                    max_new_tokens=4096,
                    max_time=1200,
                ),
            ),

            trainer_config=dict(
                trainer_args=dict(
                    num_train_epochs=10,
                    #per_device_train_batch_size=4,
                    output_dir=f"{self._path}",
                    save_strategy='no',
                    eval_strategy='no',
                    do_eval=False,
                    fp16=False,
                ),
            ),
        )

        self._config_key_order.extend(list(self._default_config.keys()))
        self._config_keys_to_exclude.extend(['dataset_df','train_df','test_df','eval_df','train_ds','eval_ds','trainer','trainwrap','llm_pipe','extractor_pipe','dataset'])

        self.update_config(self._default_config, overwrite_if_conflict=False)
        self.update_config_smart(kwargs)

        if self.is_spawned():
            llm_ref=self.llm_config
            dataset_ref=self.dataset_config
            trainer_ref=self.trainer_config
            extractor_ref=self.extractor_config


        if llm_ref:
            self.llm_pipe=LanguageModelPipeline.from_ref(llm_ref)
            self.llm_pipe.update_config(self._default_config['llm_config'])
            self.llm_pipe.update_config_smart(kwargs)
            self.llm_config=self.llm_pipe.get_config()
        
        if dataset_ref:
            self.dataset=BaseDataset.from_ref(dataset_ref)
            self.dataset_config=self.dataset.get_config()
        
        if trainer_ref:
            self.trainwrap=BaseTrainerWrap.from_ref(trainer_ref)
            self.trainwrap.update_config(self._default_config['trainer_config'])
            self.trainwrap.update_config_smart(kwargs)
            self.trainwrap.trainer_args['seed']=self.seed
            self.trainer_config=self.trainwrap.get_config()
        
        if extractor_ref:
            self.extractor_pipe=ExtractionPipeline.from_ref(extractor_ref, llm_ref=self.llm_pipe)
            self.extractor_pipe.update_config_smart(kwargs)
            self.extractor_config=self.extractor_pipe.get_config()
        
    def load_data(self, **kwargs):
        self.logger.info(f"Loading Data")

        assert hasattr(self, 'dataset'), "Dataset must be assigned before loading data"
        assert hasattr(self, 'llm_pipe'), "llm_pipe must be assigned before loading data"
        assert hasattr(self, 'extractor_pipe'), "Extraction extractor_pipe must be assigned before loading data"
        
        train_df=self.dataset.get_df_splits('train')
        eval_df=self.dataset.get_df_splits(['test','eval'])

        self.llm_pipe.load_training_tokenizer()
        self.llm_pipe.load_inference_tokenizer()

        train_df=self.extractor_pipe.prepare_df_for_finetuning(train_df)
        train_ds=self.extractor_pipe.get_ds_for_finetuning(train_df, self.llm_pipe.get_training_tokenizer()) # tokenizer type doesnt matter at this point, because padding is not applied

        eval_df=self.extractor_pipe.prepare_df_for_completion(eval_df)
        eval_ds=self.extractor_pipe.get_ds_for_completion(eval_df,  self.llm_pipe.get_inference_tokenizer())

        self.train_df=train_df
        self.eval_df=eval_df

        self.train_ds=train_ds
        self.eval_ds=eval_ds

    def get_trainer(self, model=None, **kwargs):
        assert model is not None, "model must be passed"

        self.logger.info(f"Loading Trainer")

        trainer= self.trainwrap.get_trainer(
            model=model,
            tokenizer=self.llm_pipe.get_training_tokenizer(),
            train_dataset=self.train_ds,
            #args=trainer_args,
        )

        trainer.add_callback(PredictionCallback(self))
        return trainer
    
    def evaluate_base_model(self,model):

        pred_callback=PredictionCallback(self)
        pred_df=pred_callback.eval_loop(model,self.trainer_config['trainer_args']['per_device_eval_batch_size'])
        pred_callback.save_df(pred_df,0)

    def run(self,**kwargs):

        self.logger.info(f"RUNNING")
        set_seeds(self.seed)

        if not self.is_spawned():
            self.spawn()

        if self.done:
            self.logger.info(f"Already done. Skipping.")
            return

        self.timestamp_run=get_timestamp()
        base_model=self.llm_pipe.get_model()

        self.load_data(**kwargs)

        if self.do_eval_base_model:
            self.evaluate_base_model(model=base_model)
        
        if self.do_train:
            trainer=self.get_trainer(model=base_model,**kwargs)

            self.lora_info = getattr(self.trainwrap, 'lora_info', None)

            trainer_args=trainer.args.to_dict()
            save_yaml(trainer_args, f"{self._path}/trainer_args.yaml")
            trainer.train()

        self.done=True
        self.timestamp_done=get_timestamp()
        set_config(self.get_config(),path=self._path)
        return
    
    def spawn(self):
        if not all([hasattr(self, attr) for attr in ['llm_pipe','dataset','trainwrap','extractor_pipe']]):
            raise ValueError("llm_pipe, dataset, trainwrap, extractor_pipe must be assigned before spawning.")

        if not self.llm_pipe.is_spawned(): self.llm_pipe.spawn()
        if not self.dataset.is_spawned(): self.dataset.spawn()
        if not self.trainwrap.is_spawned(): self.trainwrap.spawn()
        if not self.extractor_pipe.is_spawned(): self.extractor_pipe.spawn()
        super().spawn()

    def get_callbacks_df(self):

        fns=os.listdir(self._path)
        fns=[fn for fn in fns if fn.endswith('df.pkl')]
        callbacks_df=pd.DataFrame()
        for fn in fns:
            global_step=int(fn[:4])
            callback_df=pd.read_pickle(f"{self._path}/{fn}")
            callback_df['global_step']=global_step
            callbacks_df=pd.concat([callbacks_df,callback_df])

        callbacks_df.reset_index(drop=True, inplace=True)
        return callbacks_df



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

    