import beesup_llm
from beesup_llm import *
from beesup_llm.model_pipelines import *
from rapidfuzz import fuzz

from beesup_llm.injection import get_system_prompt

from transformers import TrainerCallback

class EvaluatorCallback(TrainerCallback):

    def __init__(self, evaluator, experiment):
        self.evaluator=evaluator
        self.experiment=experiment
        self.name=f"{evaluator.subtype}-{evaluator.id}_callback"
    
    def save_df(self, callback_df, state):
        save_path=f"{self.experiment._path}/{str(int(state.epoch))}-{str(state.global_step)}_{self.name}_df.pkl"
        callback_df.to_pickle(save_path)
        self.experiment.logger.info(f"Saved {self.name} to {save_path}")
    
    def on_epoch_end(self, args=None, state=None, control=None, **kwargs):

        if not self.evaluator.is_eval_epoch(state.epoch):
            self.experiment.logger.info(f"{self.name}\tepoch: {state.epoch}\tglobal step: {state.global_step} not an eval epoch")
            return #skip evaluation if not specified as eval epoch
            
        self.experiment.logger.info(f"epoch: {state.epoch}\tglobal step: {state.global_step}")

        model=kwargs['model']
        model.eval()

        callback_df=self.evaluator(llm_ref=model, **kwargs)
        self.save_df(callback_df, state)

class MCEEvaluatorCallback(EvaluatorCallback):
    """Multiclass Cross Entropy Loss Evaluator Callback

    fetches sample-mapped loss data from Custom Trainer Wrapper
    """

    def __init__(self, experiment):
        self.experiment=experiment
        self.name=f"mce_callback"
        self.loss_data=[]
    
    def add_loss_data(self, data):
        self.loss_data.extend(data)

    def on_epoch_end(self, args, state, control, **kwargs):

        callback_df=pd.DataFrame(self.loss_data)

        self.loss_data = []
        self.save_df(callback_df, state)
      
class Evaluator(BaseDirectory):
    """Base class for all evaluators"""
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
            n_rows=None,
            columns=[],
            eval_epochs=[],
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
            #self.llm_config=self.llm_pipe.get_config()
            #self.update_config(dict(llm_config=self.llm_pipe.get_config()), overwrite_if_conflict=False)
        
        # load source data if available
        if os.path.exists(f"{self._path}/df.pkl"):
            self.df=pd.read_pickle(f"{self._path}/df.pkl")
            self.columns=self.df.columns.tolist()
            self.logger.debug(f"Loaded df from {self._path}/df.pkl")

        # attach df if provided/ overwrite loaded df
        if df is not None: self.set_df(df)
    
    def is_eval_epoch(self, epoch):
        if self.eval_epochs==[]: return True
        elif int(epoch) in self.eval_epochs: return True
    
    def set_df(self, df=None):
        if isinstance(df, pd.DataFrame):
            self.df=df.copy()
        
        self.df.reset_index(drop=True, inplace=True)
        self.n_rows=len(self.df)
        self.columns=self.df.columns.tolist()
        return

    def spawn(self):
        super().spawn()
        if not hasattr(self, 'df'): self.logger.warning("No df provided for spawning")
        self.df.to_pickle(f"{self._path}/df.pkl")

    def get_prompt_messages(self, sample, **kwargs):
        prompt_messages=[]
        prompt_messages.append(dict(role='system', content=get_system_prompt(**kwargs)))
        prompt_messages.append(dict(role='user', content=self.get_prompt(sample, **kwargs)))
        return prompt_messages
        

    def get_pred_df(self, df=None, llm_pipe=None, **kwargs):
        if df is None: df=self.df
        pred_df=df.copy()

        pred_df['pred_completion']=[dict(info="NotImplemented")] * len(pred_df)
        return pred_df
    
    def get_eval_df(self, pred_df=None, **kwargs):
        if pred_df is None: pred_df=self.pred_df
        eval_df=pred_df.copy()

        eval_df['eval_dict']=[dict(info="NotImplemented")] * len(eval_df)
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

class MCQEvaluator(Evaluator):
    """Multiple Choice Question Evaluator"""
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
            #self.llm_config=self.llm_pipe.get_config()

    @staticmethod 
    def get_prompt(sample):
        prompt=""
        prompt+="""
You are provided with the following multiple-choice question. \
Carefully review the options and select the correct one. \
Respond only with the letter of the correct choice. \
Do not provide any additional explanation or reasoning.
""".strip()
            
        prompt+=f"\n\n### QUESTION:\n\n"
        prompt+=f"{sample.question}\n\n"

        for i, choice in enumerate(sample.choices):
            prompt+=f"{chr(65+i)}) {choice}\n"

        prompt+="\n### LETTER OF CORRECT CHOICE:\n"

        return prompt
    
    def get_pred_df(self, df=None, llm_pipe=None, **kwargs):
        if df is None: df=self.df
        pred_df=df.copy()

        pred_df['prompt_messages']=pred_df.apply(lambda x: self.get_prompt_messages(x), axis=1)
        if 'mmluidx' in df.columns: pred_df['prompt_messages']=pred_df['prompt_messages'].apply(lambda x: x[1:]) #remove system message

        pred_df=llm_pipe(pred_df)

        return pred_df
    

    @staticmethod
    def get_eval_dict(sample):

        gold_choice=chr(65+int(sample.answer)).lower()

        pred_choice=sample.pred_completion.strip().lower()
        pred_choice=re.sub(r'[^a-zA-Z]','',pred_choice)
        pred_choice=pred_choice[:1]
        
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

class QDQEvaluator(Evaluator):
    """Query Driven Questions Evaluator"""
    subtype='qdq'

    def __init__(self, ref=None, **kwargs):
        super().__init__(ref, **kwargs)

        self._default_config=dict(
            llm_config=dict(
                generation_config=dict(
                    stop_strings=['\n'],
                    #max_new_tokens=8,
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
            #self.llm_config=self.llm_pipe.get_config()
    
    @staticmethod 
    def get_prompt(sample, fewshots=None):
        prompt=""
        prompt+="""
    You are provided with a question about wild bees. \
    Your answer should consist of a semicolon-separated list of scientific names of wild bees. \
    A scientific name is always formatted as follows: <genus> <species> (<author>, <year>).\
    """.strip()
        
        prompt+=f"\n\n### EXAMPLE:\n"
        for _,fewshot in fewshots.iterrows():
            prompt+=f"Question: {fewshot.question}\n"
            prompt+=f"Answer: {'; '.join(fewshot.gold_items)}\n"

        prompt+="\n\n"
        prompt+=f"Question: {sample.question}\n"
        prompt+=f"Answer:"

        return prompt
    
    def get_pred_df(self, df=None, llm_pipe=None, **kwargs):
        if df is None: df=self.df

        fewshots_df=df[:1].copy()
        pred_df=df[1:].reset_index(drop=True).copy()

        pred_df['prompt_messages']=pred_df.apply(lambda x: self.get_prompt_messages(x, fewshots=fewshots_df), axis=1)

        #pred_df['prompt_messages']=pred_df.apply(lambda x: [dict(role='user', content=self.get_prompt(x, fewshots_df))], axis=1)

        max_new_tokens=int(max([llm_pipe.count_tokens("; ".join(sample.gold_items)) for _,sample in pred_df.iterrows()])*2) #allow maximum twice the number of tokens in the gold items
        self.logger.info(f"max_new_tokens: {max_new_tokens}")

        pred_df=llm_pipe(pred_df, max_new_tokens=max_new_tokens)

        return pred_df


    @staticmethod
    def get_eval_dict(sample):

        def normalize(item_str):
            item_str=item_str.strip()
            item_str=item_str.lower()
            item_str=re.sub(r'\s+', ' ', item_str)
            return item_str
        
        gold_df=pd.DataFrame(sample.gold_items, columns=['gold_item']).reset_index(names=['g'])
        gold_df['gold_val']=gold_df['gold_item'].apply(normalize)

        pred_items=list(set([c.strip() for c in re.split(r';',sample.pred_completion)]))
        pred_df=pd.DataFrame(pred_items, columns=['pred_item']).reset_index(names=['p'])
        pred_df['pred_val']=pred_df['pred_item'].apply(normalize)

        match_data=[]
        for g, gold_row in gold_df.iterrows():
            for p, pred_row in pred_df.iterrows():
                match_row=dict(
                    g=g,
                    p=p,
                    gold_val=gold_row['gold_val'],
                    pred_val=pred_row['pred_val'],
                    fuzz_ratio=fuzz.ratio(gold_row['gold_val'],pred_row['pred_val'])/100,
                )
                match_data.append(match_row)

        match_df=pd.DataFrame(match_data)
        match_df.sort_values(by='fuzz_ratio',ascending=False,inplace=True)
        match_df.reset_index(drop=True,inplace=True)

        umap_df=match_df.copy()

        i=0
        while True:

            g_is_unique=umap_df[:i+1]['g'].is_unique
            p_is_unique=umap_df[:i+1]['p'].is_unique

            if not (g_is_unique) & (p_is_unique):
                umap_df.drop(umap_df.index[i],inplace=True)
                continue

            if i==len(umap_df): break
            if i > 10000: break
            i+=1

        tp_df=umap_df[umap_df.fuzz_ratio>0.9].copy()
        fp_df=pred_df[~pred_df.p.isin(tp_df.p)].copy()
        fn_df=gold_df[~gold_df.g.isin(tp_df.g)].copy()

        tp=len(tp_df)
        tp_fuzzy=tp_df.fuzz_ratio.sum()
        fp=len(fp_df)
        fn=len(fn_df)

        eval_dict=dict(
            tp=tp, tp_fuzzy=tp_fuzzy, fp=fp, fn=fn,
            #P=precision(tp=tp,fp=fp,fn=fn,tp_fuzzy=tp_fuzzy),
            #R=recall(tp=tp,fp=fp,fn=fn, tp_fuzzy=tp_fuzzy),
            #F1=f1_score(tp=tp,fp=fp,fn=fn, tp_fuzzy=tp_fuzzy),
        )

        return eval_dict

    # def get_eval_df(self, pred_df=None, **kwargs):
    #     if pred_df is None: pred_df=self.pred_df

    #     eval_df=pred_df.copy()
    #     eval_df['eval_dict']=eval_df.apply(self.get_eval_dict, axis=1)

    #     return eval_df

class FFQEvaluator(Evaluator):
    """Free Form Question Evaluator"""
    subtype='ffq'

    def __init__(self, ref=None, **kwargs):
        super().__init__(ref, **kwargs)

        self._default_config=dict(
            llm_config=dict(
                generation_config=dict(
                    stop_strings=['\n'],
                    max_new_tokens=500,
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
            #self.llm_config=self.llm_pipe.get_config()

    @staticmethod 
    def get_prompt(sample):
        return sample['question']
    

    def get_pred_df(self, df=None, llm_pipe=None, **kwargs):
        if df is None: df=self.df
        pred_df=df.copy()

        pred_df['prompt_messages']=pred_df.apply(lambda x: self.get_prompt_messages(x), axis=1)
        pred_df=llm_pipe(pred_df)

        return pred_df

class KRQEvaluator(Evaluator):
    """Key Response Question Evaluator"""
    subtype='krq'
