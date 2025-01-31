import beesup_llm
from beesup_llm import *
from beesup_llm.toolkit.setup_utils import *
from beesup_llm.toolkit.llm_utils import *
from beesup_llm.toolkit.system import *

from beesup_llm.dataset import BaseDataset
from beesup_llm.training import *

from beesup_llm.model_pipelines import *

from transformers import TrainerCallback

#class PredictionCallback(TrainerCallback):



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

    def __init__(self, ref=None, dataset_ref=None, llm_ref=None, trainer_ref=None, **kwargs):
        super().__init__(ref, **kwargs)

        self._default_config=dict(
            done = False,
            seed = 55,
            do_eval_base_model=True,
            do_mcq_eval=True,
            do_qdq_eval=True,
            do_ffq_eval=True,
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