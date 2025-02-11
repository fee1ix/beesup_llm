import beesup_llm
from beesup_llm import *
from beesup_llm.toolkit.setup_utils import *
from beesup_llm.toolkit.llm_utils import *
from beesup_llm.model_pipelines import *
from beesup_llm.toolkit.system import *


from beesup_llm.dataset import BaseDataset



class BaseExperiment(BaseDirectory):
    type='llm_experiment'

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


    def __init__(self, ref=None, dataset_ref=None, llm_ref=None, eval_refs=[], **kwargs):
        super().__init__(ref, **kwargs)

        self._default_config=dict(
            done = False,
            seed = 55,

            llm_config=dict(
                generation_config=dict(
                    max_new_tokens=4096,
                    max_time=1200,
                ),
            ),
            eval_configs=[]
        )    


        self._config_key_order.extend(list(self._default_config.keys()))
        self._config_keys_to_exclude.extend(['dataset','dataset_df','llm_pipe','eval_pipes'])

        self.update_config(self._default_config, overwrite_if_conflict=False)
        self.update_config_smart(kwargs)

        if self.is_spawned():
            llm_ref=self.llm_config
            dataset_ref=self.dataset_config
            eval_refs=self.eval_configs


        if llm_ref:
            self.llm_pipe=LanguageModelPipeline.from_ref(llm_ref)
            self.llm_pipe.update_config(self._default_config['llm_config'])
            self.llm_pipe.update_config_smart(kwargs)
            self.llm_config=self.llm_pipe.get_config()
        
        if dataset_ref:
            self.dataset=BaseDataset.from_ref(dataset_ref)
            self.dataset_config=self.dataset.get_config()
        
        
        if eval_refs:
            self.evaluators=[LLMEvaluator.from_ref(eval_ref) for eval_ref in eval_refs]
            self.eval_configs=[pipe.get_config() for pipe in self.evaluators]







