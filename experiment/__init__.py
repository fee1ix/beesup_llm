import beesup_llm
from beesup_llm import *
from beesup_llm.toolkit.setup_utils import *
from beesup_llm.toolkit.llm_utils import *
from beesup_llm.model_pipelines import *
from beesup_llm.toolkit.system import *

from beesup_llm.dataset import BaseDataset



class BaseEvaluator(BaseDirectory):
    type='llm_evaluator'

    @classmethod
    def from_ref(cls, ref=None, **kwargs):
        kwargs.update(get_cls_attrs(cls))
        cls.logger.debug(f"{cls} ref={ref}, kwargs = {kwargs}\n")
        pre_config = get_config_from_ref(ref, **kwargs)

        if MCQEvaluator.matches(pre_config): return MCQEvaluator(ref=pre_config, **kwargs)
        if QDQEvaluator.matches(pre_config): return QDQEvaluator(ref=pre_config, **kwargs)
        if FFQEvaluator.matches(pre_config): return FFQEvaluator(ref=pre_config, **kwargs)
        if KRQEvaluator.matches(pre_config): return KRQEvaluator(ref=pre_config, **kwargs)

        return cls(ref=pre_config, **kwargs)

    @classmethod
    def matches(cls, ref):
        if getattr_or_key(ref, 'subtype') == cls.subtype: return True
        return False

    def __init__(self, ref=None, llm_ref=None, df=None,**kwargs):
        super().__init__(ref, **kwargs)

        self._default_config=dict(
            subtype=None,
            remarks=None,
            llm_config=dict(
            ) 
        )

        self._config_key_order.extend([k for k in self._default_config.keys() if k not in self._config_key_order])
        self._config_keys_to_exclude.extend(['df'])

        self.update_config(self._default_config, overwrite_if_conflict=False)
        self.update_config_smart(kwargs)

        if llm_ref:
            self.llm_pipe=LanguageModelPipeline.from_ref(llm_ref)
            self.llm_pipe.update_config(self.llm_config)
            #self.llm_pipe.update_config(self._default_config['llm_config'])
            self.llm_pipe.update_config_smart(kwargs)
            self.llm_config=self.llm_pipe.get_config()
            #self.update_config(dict(llm_config=self.llm_pipe.get_config()), overwrite_if_conflict=False)
        

        # load source data if available
        if os.path.exists(f"{self._path}/df.pkl"):
            self.df=pd.read_pickle(f"{self._path}/df.pkl")
            self.logger.debug(f"Loaded df from {self._path}/df.pkl")

        # attach df if provided/ overwrite loaded df

        if isinstance(df, pd.DataFrame):
            self.df=df
            self.logger.debug(f"Attached df from input")

    def spawn(self):
        super().spawn()
        if not hasattr(self, 'df'): self.logger.warning("No df provided for spawning")
        self.df.to_pickle(f"{self._path}/df.pkl")

    def get_pred_df(self, df=None, llm_pipe=None, **kwargs):
        if df is None: df=self.df
        pred_df=df.copy()

        pred_df['pred_completion']='NotImplemented'
        return pred_df
    
    def get_eval_df(self, pred_df=None, **kwargs):
        if pred_df is None: pred_df=self.pred_df
        eval_df=pred_df.copy()

        eval_df['eval_dict']='NotImplemented'
        return eval_df


    def __call__(self, llm_ref=None, **kwargs):

        if llm_ref:
            llm_pipe=LanguageModelPipeline.from_ref(llm_ref)
            llm_pipe.update_config(self.llm_config)
            
        elif hasattr(self, 'llm_pipe'):
            llm_pipe=self.llm_pipe
        
    
        #PREDICT
        pred_df=self.df.copy()
        try:
            pred_df=self.get_pred_df(pred_df, llm_pipe, **kwargs)

        except Exception as e:
            pred_df['pred_completion']=f"PredictionError: {e}"
            self.logger.info(f"PredictionError: {e}")
        self.pred_df=pred_df


        #EVALUATE
        eval_df=self.pred_df.copy()
        try:
            eval_df=self.get_eval_df(pred_df=eval_df, **kwargs)
        except Exception as e:
            eval_df['eval_dict']=f"EvaluationError: {e}"
            self.logger.info(f"EvaluationError: {e}")
        self.eval_df=eval_df
            
        return eval_df

    
class MCQEvaluator(BaseEvaluator):
    subtype='mcq'

    def __init__(self, ref=None, **kwargs):
        super().__init__(ref, **kwargs)

        self._default_config=dict(
            llm_config=dict(
                generation_config=dict(
                    stop_strings=['\n'],
                    max_new_tokens=8,
                )
            )
        )

        self.update_config(self._default_config, overwrite_if_conflict=True)
        self.update_config_smart(
            kwargs, 
            interpret_none_as_val=True, 
            overwrite_if_conflict=True, 
            allow_new_atomic_keys=False, 
            allow_new_nested_keys=False
        )

        if hasattr(self, 'llm_pipe'):
            self.llm_pipe.update_config(self.llm_config)
            self.llm_pipe.update_config_smart(kwargs)
            self.llm_config=self.llm_pipe.get_config()

    @staticmethod 
    def get_prompt(sample):
        prompt=""
        prompt+="""
You are provided with the following multiple-choice question. \
Carefully review the options and select the correct one. \
Respond only with the capital letter of the correct choice. \
Do not provide any additional explanation or reasoning.
""".strip()
            
        prompt+=f"\n\n### QUESTION:\n\n"
        prompt+=f"{sample.question}\n\n"

        for i, choice in enumerate(sample.choices):
            prompt+=f"{chr(65+i)}) {choice}\n"

        prompt+="\n### LETTER OF CORRECT CHOICE:\n"

        return [dict(role='user', content=prompt)]
    
    def get_pred_df(self, df=None, llm_pipe=None, **kwargs):
        if df is None: df=self.df
        pred_df=df.copy()

        pred_df['prompt_messages']=pred_df.apply(self.get_prompt, axis=1)
        pred_df=llm_pipe(pred_df)

        return pred_df
    

    @staticmethod
    def get_eval_dict(sample):

        gold_choice=chr(65+sample.answer).lower()

        pred_choice=sample.pred_completion.strip().lower()
        pred_choice=re.sub(r'[^a-zA-Z]','',pred_choice)
        
        eval_dict=dict(
            pred_choice	= pred_choice,
            gold_choice	= gold_choice,
            tp = 1 if gold_choice==pred_choice else 0
        )

        return eval_dict
    

    def get_eval_df(self, pred_df=None, **kwargs):
        if pred_df is None: pred_df=self.pred_df

        eval_df=pred_df.copy()
        eval_df['eval_dict']=eval_df.apply(self.get_eval_dict, axis=1)

        return eval_df





class QDQEvaluator(BaseEvaluator):
    subtype='qdq'

class FFQEvaluator(BaseEvaluator):
    subtype='ffq'

class KRQEvaluator(BaseEvaluator):
    subtype='krq'




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







