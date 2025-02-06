from beesup_llm import *
from beesup_llm.toolkit.setup_utils import *
from beesup_llm.injection.evaluation_utils import *

from beesup_llm.dataset import *
from beesup_llm.model_pipelines import *

import pickle
import pandas as pd


class EvaluationPipeline(BaseDirectory):
    type='evaluation_pipeline'

    @classmethod
    def from_ref(cls, ref=None, **kwargs):
        kwargs.update(get_cls_attrs(cls))
        cls.logger.debug(f"{cls} ref={ref}, kwargs = {kwargs}\n")
        pre_config = get_config_from_ref(ref, **kwargs)

        if MCQEvaluationPipeline.matches(pre_config): return MCQEvaluationPipeline(ref=pre_config, **kwargs)
        if FFQEvaluationPipeline.matches(pre_config): return FFQEvaluationPipeline(ref=pre_config, **kwargs)
        if QDQEvaluationPipeline.matches(pre_config): return QDQEvaluationPipeline(ref=pre_config, **kwargs)


        return cls(ref=pre_config, **kwargs)

    def __init__(self, ref=None, llm_ref=None, df=None,**kwargs):
        super().__init__(ref, **kwargs)

        self._default_config=dict(
            subtype=None,
            llm_config=dict(
                generation_config=dict(
                    stop_strings=[],
                )
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
        
        if self.is_spawned(): self.load()

        if isinstance(df, pd.DataFrame): self.df=df

    
    def load(self):
        if os.path.exists(f"{self._path}/df.pkl"):
            self.df=pd.read_pickle(f"{self._path}/df.pkl")
            self.logger.debug(f"Loaded df from {self._path}/df.pkl")

    
    def spawn(self):

        super().spawn()

        if not hasattr(self, 'df'): self.logger.warning("No df provided for spawning")
        self.df.to_pickle(f"{self._path}/df.pkl")

        
    def get_eval_df(self, llm_pipe, **kwargs):

        eval_df=self.df.copy()
        eval_df=llm_pipe(eval_df, use_chatformat=True, stop_strings=['\n'])

        return eval_df


    def __call__(self, llm_ref=None, **kwargs):
    
        if llm_ref:
            llm_pipe=LanguageModelPipeline.from_ref(llm_ref)
            llm_pipe.update_config(self.llm_config)
            #llm_pipe.update_config(self._default_config['llm_config'])
    
        elif hasattr(self, 'llm_pipe'):
            llm_pipe=self.llm_pipe

        try:
            eval_df=self.get_eval_df(llm_pipe, **kwargs)
        except Exception as e:
            eval_df=self.df.copy()
            eval_df['pred_completion']=f"Error in EvaluationPipeline: {e}"
            self.logger.info(f"Error in EvaluationPipeline: {e}")
            
        return eval_df
        
class FFQEvaluationPipeline(EvaluationPipeline):

    @staticmethod
    def matches(ref):
        if getattr_or_key(ref, 'subtype') in ['ffq']: return True
        return False

    def __init__(self, ref=None, **kwargs):
        super().__init__(ref, **kwargs)

        self._default_config=dict(
            subtype='ffq',
            llm_config=dict(
                generation_config=dict(
                    max_new_tokens=500,
                )
            )
        )

        self._config_key_order.extend(list(self._default_config.keys()))
        self._config_keys_to_exclude.extend([])
        
        self.update_config(self._default_config, overwrite_if_conflict=True)
        self.update_config_smart(
            kwargs, 
            interpret_none_as_val=True, 
            overwrite_if_conflict=True, 
            allow_new_atomic_keys=False, 
            allow_new_nested_keys=False
        )


class MCQEvaluationPipeline(EvaluationPipeline):

    @staticmethod
    def matches(ref):
        if getattr_or_key(ref, 'subtype') in ['mcq']: return True
        return False

    def __init__(self, ref=None, **kwargs):
        super().__init__(ref, **kwargs)

        self._default_config=dict(
            subtype='mcq',
            llm_config=dict(
                generation_config=dict(
                    stop_strings=['\n'],
                    max_new_tokens=15,
                )
            )
        )

        self._config_key_order.extend(list(self._default_config.keys()))
        self._config_keys_to_exclude.extend([])
        
        self.update_config(self._default_config, overwrite_if_conflict=True)
        self.update_config_smart(
            kwargs, 
            interpret_none_as_val=True, 
            overwrite_if_conflict=True, 
            allow_new_atomic_keys=False, 
            allow_new_nested_keys=False
        )

    @staticmethod 
    def get_prompt(sample):
        prompt=""
        prompt+="""
    You are provided with the following multiple-choice question. \
    Carefully review the options and select the correct one/ the correct ones. \
    Respond only with the capital letter or a concatenation of correct letters. \
    Do not provide any additional explanation or reasoning.
    """.strip()

        prompt+=f"\n\n### QUESTION:\n{sample.question}\n\n"
        prompt+="### LETTER/ LETTERS OF CORRECT CHOICE/ CHOICES:\n"
        return prompt
    
    @staticmethod
    def get_eval_dict(sample):

        gold_choices=set(sample['right_choices'])
        pred_choices=set([a.strip().upper() for a in re.split(r'[,\?;! ]',sample['pred_completion']) if (a.strip() and a in LETTERS)]) 

        choices=set(gold_choices.union(set(sample['wrong_choices'])))

        tp=len(gold_choices.intersection(pred_choices))
        fp=len(pred_choices-gold_choices)
        fn=len(gold_choices-pred_choices)
        tn=len(choices-(gold_choices.union(pred_choices)))

        pre=tp/(tp+fp) if tp+fp>0 else 0
        rec=tp/(tp+fn) if tp+fn>0 else 0
        f1_score=2*(pre*rec)/(pre+rec) if pre+rec>0 else 0

        eval_dict=dict(
            tp=tp, fp=fp, fn=fn, tn=tn,
            P=pre, R=rec, F1=f1_score
        )

        return eval_dict
    
    @staticmethod
    def get_completion(sample, llm_pipe):
            completion=llm_pipe(sample.prompt, use_chatformat=True, stream=False, stop_strings=['\n'])[0]['generated_text']
            completion=clean_completion(completion)
            completion=completion.replace('<|eot_id|>','')
            return completion

    
    def get_eval_df(self, llm_pipe, **kwargs):


        eval_df=self.df.copy()

        eval_df['prompt']=eval_df.apply(self.get_prompt, axis=1)
        eval_df['pred_completion']=eval_df.apply(lambda x: self.get_completion(x, llm_pipe), axis=1)

        eval_df['eval_dict']=eval_df.apply(self.get_eval_dict, axis=1)
        eval_df = eval_df.join(pd.json_normalize(eval_df['eval_dict']))

        return eval_df

class QDQEvaluationPipeline(EvaluationPipeline):
 
    @staticmethod
    def matches(ref):
        if getattr_or_key(ref, 'subtype') in ['qdq']: return True
        return False

    def __init__(self, ref=None, **kwargs):
        super().__init__(ref, **kwargs)

        self._default_config=dict(
            subtype='qdq',
            llm_config=dict(
                generation_config=dict(
                    stop_strings=['\n'],
                )
            )
        )

        self._config_key_order.extend(list(self._default_config.keys()))
        self._config_keys_to_exclude.extend([])
        
        self.update_config(self._default_config, overwrite_if_conflict=True)
        self.update_config_smart(
            kwargs, 
            interpret_none_as_val=True, 
            overwrite_if_conflict=True, 
            allow_new_atomic_keys=False, 
            allow_new_nested_keys=False
        )


    @staticmethod 
    def get_prompt(sample, fewshots):
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
    
    @staticmethod
    def get_eval_dict(sample):

        def normalize(item_str):
            item_str=item_str.strip()
            item_str=item_str.lower()
            item_str=re.sub(r'\s+', ' ', item_str)
            return item_str
        
        gold_df=pd.DataFrame(sample.gold_items, columns=['gold_item']).reset_index(names=['g'])
        gold_df['gold_val']=gold_df['gold_item'].apply(normalize)


        pred_df=pd.DataFrame(sample.pred_items, columns=['pred_item']).reset_index(names=['p'])
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
            P=precision(tp=tp,fp=fp,fn=fn,tp_fuzzy=tp_fuzzy),
            R=recall(tp=tp,fp=fp,fn=fn, tp_fuzzy=tp_fuzzy),
            F1=f1_score(tp=tp,fp=fp,fn=fn, tp_fuzzy=tp_fuzzy),
        )

        return eval_dict

    def get_eval_df(self, llm_pipe, **kwargs):
    

        if hasattr(self,'df'):
            self.fewshots_df=self.df[:1].copy() #use first sample as fewshot
            self.samples_df=self.df[1:].reset_index(drop=True).copy()

        eval_df=self.samples_df.copy()
        eval_df['prompt']=eval_df.apply(lambda x: self.get_prompt(x, self.fewshots_df), axis=1)

        max_new_tokens=int(max([llm_pipe.count_tokens("; ".join(sample.gold_items)) for _,sample in eval_df.iterrows()])*2) #allow maximum twice the number of tokens in the gold items
        self.logger.info(f"max_new_tokens: {max_new_tokens}")

        eval_df=llm_pipe(eval_df, use_chatformat=False, stop_strings=['\n'], max_new_tokens=max_new_tokens)

        eval_df['pred_items']=eval_df['pred_completion'].apply(lambda x: list(set([p.strip() for p in re.split(r';',x)])))
        eval_df['eval_dict']=eval_df.apply(self.get_eval_dict, axis=1)
        eval_df = eval_df.join(pd.json_normalize(eval_df['eval_dict']))

        return eval_df
    

        


