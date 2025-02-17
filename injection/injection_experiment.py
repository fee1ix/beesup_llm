import beesup_llm
from beesup_llm import *
from beesup_llm.toolkit.setup_utils import *
from beesup_llm.toolkit.llm_utils import *
from beesup_llm.toolkit.system import *

from beesup_llm.dataset import BaseDataset
#from beesup_llm.training import *

from beesup_llm.finetuning_pipelines import *

from beesup_llm.model_pipelines import *

from beesup_llm.experiment import *
from beesup_llm.injection.evaluator import *

from datasets import Dataset
from transformers import TrainerCallback, TrainerState



class LogHistoryCallback(TrainerCallback):
    def __init__(self, experiment):
        self.experiment=experiment

    def on_epoch_end(self, args=None, state=None, control=None, **kwargs):
        log_history_df=pd.DataFrame(state.log_history)
        log_history_df.to_pickle(f"{self.experiment._path}/log_history_df.pkl")





class InjectionExperiment(BaseDirectory):
    type='injection_experiment'

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

    def __init__(self, ref=None, dataset_ref=None, llm_ref=None, ftn_ref=None, eval_refs=[], **kwargs):
        super().__init__(ref, **kwargs)

        self._default_config=dict(
            done = False,
            seed = 55,
            do_eval_base_model=True,
            do_finetuning=True,
            remarks='',
            llm_config=dict(
                generation_config=dict(
                    max_new_tokens=4096,
                    max_time=1200,
                ),
            ),
            ftn_config=dict(
                trainer_config=dict(
                    #num_train_epochs=18,
                    #per_device_train_batch_size=4,
                    output_dir=f"{self._path}",
                    save_strategy='no',
                    eval_strategy='no',
                    do_eval=False,
                    fp16=False,
                ),
            ),
            eval_configs=[]
        )    


        self._config_key_order.extend(list(self._default_config.keys()))
        self._config_keys_to_exclude.extend(['dataset_df','train_df','test_df','eval_df','train_ds','eval_ds','trainer','ftn_pipe','llm_pipe','dataset','evaluators'])

        self.update_config(self._default_config, overwrite_if_conflict=False)
        self.update_config_smart(kwargs)

        if self.is_spawned():
            if llm_ref==None: llm_ref=self.llm_config
            if ftn_ref==None: ftn_ref=self.ftn_config
            dataset_ref=self.dataset_config
            eval_refs=self.eval_configs

        if llm_ref:
            self.llm_pipe=LanguageModelPipeline.from_ref(llm_ref)
            self.llm_pipe.update_config(self._default_config['llm_config'])
            self.llm_pipe.update_config_smart(kwargs)
            self.llm_config=self.llm_pipe.get_config()

        if ftn_ref:
            self.ftn_pipe=FinetuningPipeline.from_ref(ftn_ref, llm_ref=self.llm_pipe)
            self.ftn_pipe.update_config(self._default_config['ftn_config'], overwrite_if_conflict=True)
            self.ftn_pipe.update_config_smart(kwargs, overwrite_if_conflict=True)
            self.ftn_pipe.trainer_config['seed']=self.seed
            self.ftn_config=self.ftn_pipe.get_config()

        
        if dataset_ref:
            self.dataset=BaseDataset.from_ref(dataset_ref)
            self.dataset_config=self.dataset.get_config()
        
        self.evaluators=[]
        if eval_refs:
            self.evaluators=[Evaluator.from_ref(eval_ref) for eval_ref in eval_refs]
            self.eval_configs=[pipe.get_config() for pipe in self.evaluators]

    def load_data(self, **kwargs):
        self.logger.info(f"Loading Data")
    
        assert hasattr(self, 'dataset'), "Dataset must be assigned before loading data"
        assert hasattr(self, 'llm_pipe'), "llm_pipe must be assigned before loading data"
        
        train_df=self.dataset.get_df_splits('train')
        self.train_df=train_df

        self.llm_pipe.load_training_tokenizer()
        self.llm_pipe.load_inference_tokenizer()

        train_ds=Dataset.from_list(train_df.apply(lambda x: prepare_sample_for_chat_finetuning(x, self.llm_pipe.get_training_tokenizer(), use_as_id='kidx'),axis=1).to_list())
        self.train_ds=train_ds

    def run(self,**kwargs):

        self.logger.info(f"RUNNING")
        set_seeds(self.seed)

        if not self.is_spawned(): self.spawn()

        if self.done:
            self.logger.info(f"Already done. Skipping.")
            return
        
        self.timestamp_run=get_timestamp()
        self.load_data(**kwargs)
        self.llm_pipe.prepare_inference()

        if self.do_eval_base_model:
            for evaluator in self.evaluators:
                eval_callback=EvaluatorCallback(evaluator, self)
                eval_callback.on_epoch_end(state=TrainerState(epoch=0), model=self.llm_pipe.model)

        if self.do_finetuning:
            self.ftn_pipe.load_trainer(
                model=self.llm_pipe.model,
                train_dataset=self.train_ds,
            )

            self.ftn_pipe.trainer.add_callback(LogHistoryCallback(self))
            self.ftn_pipe.trainer.add_callback(MCEEvaluatorCallback(self))
            for evaluator in self.evaluators:
                self.ftn_pipe.trainer.add_callback(EvaluatorCallback(evaluator, self))

            self.lora_info=getattr(self.ftn_pipe, 'lora_info', None)
            save_yaml(self.ftn_pipe.trainer.args.to_dict(), f"{self._path}/trainer_args.yaml")
            set_config(self.get_config(),path=self._path)

            self.ftn_pipe.trainer.train()
        

        self.done=True
        self.timestamp_done=get_timestamp()
        set_config(self.get_config(),path=self._path)
        return

    def spawn(self):
        if not all([hasattr(self, attr) for attr in ['llm_pipe','dataset','ftn_pipe']]):
            raise ValueError("llm_pipe, dataset, ftn_pipe must be assigned before spawning.")

        if not self.llm_pipe.is_spawned(): self.llm_pipe.spawn()
        if not self.dataset.is_spawned(): self.dataset.spawn()
        if not self.ftn_pipe.is_spawned(): self.ftn_pipe.spawn()


        for evaluator in self.evaluators:
            if not evaluator.is_spawned(): self.logger.warning(f"{evaluator} not spawned")

        super().spawn()

    def get_callbacks_df(self):

        fns=os.listdir(self._path)
        fns=[fn for fn in fns if fn.endswith('callback_df.pkl')]
        callbacks_df=pd.DataFrame()
        for fn in fns:
            try:
                epoch,global_step,evaluator_type,evaluator_id=re.match(r'(\d+)-(\d+)_([a-z]+)(?:-(\d+))?.*',fn).groups()
                epoch,global_step=int(epoch),int(global_step)
                if evaluator_id: evaluator_id=int(evaluator_id)

            except:
                self.logger.warning(f"Could not regex-parse {fn}")
                epoch,global_step,evaluator_type,evaluator_id=None,None,None,None

            callback_df=pd.read_pickle(f"{self._path}/{fn}")

            if 'epoch' not in callback_df.columns:
                callback_df['epoch']=epoch

            if 'global_step' not in callback_df.columns:
                callback_df['global_step']=global_step

            callback_df['evaluator_type']=evaluator_type
            callback_df['evaluator_id']=evaluator_id
            callbacks_df=pd.concat([callbacks_df,callback_df])
        callbacks_df.reset_index(drop=True, inplace=True)

        return callbacks_df

    def get_log_history_df(self):
        return pd.read_pickle(f"{self._path}/log_history_df.pkl")







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
    experiment = InjectionExperiment(experiment_dir)
    experiment.run()
        
