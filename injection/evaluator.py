import beesup_llm
from beesup_llm import *
from beesup_llm.model_pipelines import *

from beesup_llm.injection import *
from beesup_llm.injection.taxomizer import *
from beesup_llm.injection.rag_pipeline import RAGPipeline
from beesup_llm.injection.injection_experiment import InjectionExperiment


from rapidfuzz import fuzz

from transformers import TrainerCallback

class EvaluatorCallback(TrainerCallback):

    def __init__(
            self,
            evaluator,
            experiment: InjectionExperiment,
            rag_pipe: RAGPipeline = None
            ):
        
        self.evaluator=evaluator
        self.experiment=experiment
        self.rag_pipe=rag_pipe

        self.name=f"{evaluator.subtype}:{evaluator.id}"

        if self.rag_pipe:
            self.name+=f"-rag:{rag_pipe.id}"
            
    def save_callback_df(self, callback_df:pd.DataFrame, epoch=0, global_step=0, **kwargs):

        epoch, global_step = int(epoch), int(global_step)

        save_path=f"{self.experiment._path}/{epoch}:{global_step}_{self.name}_callback_df.pkl"
        callback_df.to_pickle(save_path)
        self.experiment.logger.info(f"Saved {self.name} to {save_path}")
    
    def on_epoch_end(self, args=None, state=None, control=None, **kwargs):

        if not self.evaluator.is_eval_epoch(state.epoch):
            self.experiment.logger.info(f"{self.name}\tepoch: {state.epoch}\tglobal step: {state.global_step} not an eval epoch")
            return #skip evaluation if not specified as eval epoch
        
        toc=None
        if hasattr(self.experiment,'taxomizer'):
            toc=self.experiment.taxomizer.get_table_of_contents()

        self.experiment.logger.info(f"epoch: {state.epoch}\tglobal step: {state.global_step}")

        model=kwargs['model']
        model.eval()

        if self.rag_pipe:
            self.rag_pipe.add_ranking_df(self.evaluator.df, **kwargs)
            self.rag_pipe.add_briefing_df(self.evaluator.df, **kwargs)
            

        callback_df=self.evaluator(llm_ref=model, toc=toc, **kwargs)
        self.save_callback_df(callback_df, **state.__dict__)
        self.evaluator.load_df() #reload evaluator df to remove ranking and briefing columns

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
    
        return cls(ref=pre_config, **kwargs)

    @classmethod
    def matches(cls, ref):
        if getattr_or_key(ref, 'subtype') == cls.subtype: return True
        return False

    def __init__(self, ref=None, llm_ref=None, df=pd.DataFrame(), **kwargs):
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
            self.llm_pipe.update_config_smart(kwargs)

        self.load_df()

        # attach df if provided/ overwrite loaded df
        if not df.empty:
            self.df=df

        self.set_df()
        self.df=self.df.iloc[:7].copy()

    def load_df(self):
        # load source data if available
        if os.path.exists(f"{self._path}/df.pkl"):
            self.df=pd.read_pickle(f"{self._path}/df.pkl")
            self.columns=self.df.columns.tolist()
            self.logger.debug(f"Loaded df from {self._path}/df.pkl")
            self.set_df()

    def set_df(self):   
        self.df.reset_index(drop=True, inplace=True)
        self.n_rows=len(self.df)
        self.columns=self.df.columns.tolist()
        return

    def is_eval_epoch(self, epoch):
        if self.eval_epochs==[]: return True
        elif int(epoch) in self.eval_epochs: return True
    


    def spawn(self):
        super().spawn()
        if not hasattr(self, 'df'): self.logger.warning("No df provided for spawning")
        self.df.to_pickle(f"{self._path}/df.pkl")

    def get_prompt_messages(self, sample, **kwargs):

        prompt_messages=[]
        prompt_messages.append(dict(role='system', content=get_system_prompt(**sample, **kwargs)))
        prompt_messages.append(dict(role='user', content=self.get_prompt(**sample, **kwargs)))
        return prompt_messages
        
    def get_pipe_df(self, df = pd.DataFrame(), **kwargs):
        if df.empty: df=self.df

        pipe_df=df.copy()
        pipe_df['prompt_messages']=pipe_df.apply(lambda x: self.get_prompt_messages(x, **kwargs), axis=1)
        return pipe_df
    
    def add_pred_completion(self, pipe_df: pd.DataFrame, llm_ref=None, **kwargs):
        
        if llm_ref:
            llm_pipe=LanguageModelPipeline.from_ref(llm_ref, **kwargs)
            llm_pipe.update_config(self.llm_config)

        pipe_df=llm_pipe.add_pred_completion(pipe_df, **kwargs)

    @staticmethod
    def get_eval_dict(**kwargs):
        return dict(info="NotImplemented")
    
    def add_eval_dict(self, pipe_df: pd.DataFrame, **kwargs):
        assert 'pred_completion' in pipe_df.columns, "pred_completion missing in pipe_df"
        pipe_df['eval_dict']=pipe_df.apply(lambda x: self.get_eval_dict(**x, **kwargs), axis=1)


    def __call__(self, llm_ref=None, **kwargs):
    
        pipe_df=self.get_pipe_df(**kwargs)
        self.add_pred_completion(pipe_df, llm_ref=llm_ref, **kwargs)
        self.add_eval_dict(pipe_df, **kwargs)
                
        return pipe_df

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


    def get_pipe_df(self, pipe_df = pd.DataFrame(), **kwargs):
        if pipe_df.empty: pipe_df=self.df.copy()

        assert 'question' in pipe_df.columns, "question missing in df"
        assert 'choices' in pipe_df.columns, "choices missing in df"

        if 'mmluidx' in pipe_df.columns:
            #neglect system message if MMLU sample, don't pass kwargs!
            pipe_df.drop(columns=['briefing_df'], inplace=True, errors='ignore')
            pipe_df['prompt_messages']=pipe_df.apply(lambda x: self.get_prompt_messages(x)[1:], axis=1)
        
        else:
            pipe_df['prompt_messages']=pipe_df.apply(lambda x: self.get_prompt_messages(x, **kwargs), axis=1)
        
        return pipe_df

    @staticmethod 
    def get_prompt(sample=dict(), **kwargs):
        return get_mcq_prompt(**sample, **kwargs)

    @staticmethod
    def get_eval_dict(pred_completion, answer, **kwargs):

        gold_choice=chr(65+int(answer)).lower()

        pred_choice=pred_completion.strip().lower()
        pred_choice=re.sub(r'[^a-zA-Z]','',pred_choice)
        pred_choice=pred_choice[:1]
        
        eval_dict=dict(
            pred_choice	= pred_choice,
            gold_choice	= gold_choice,
            tp = 1 if gold_choice==pred_choice else 0
        )

        return eval_dict
    

class QDQEvaluator(Evaluator):
    """Query Driven Questions Evaluator"""
    subtype='qdq'

    def __init__(self, ref=None, **kwargs):
        super().__init__(ref, **kwargs)

        self._default_config=dict(
            n_fewshots=3,
            llm_config=dict(
                generation_config=dict(
                    stop_strings=['\n'],
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

    def get_pipe_df(self, pipe_df = pd.DataFrame(), **kwargs):
        if pipe_df.empty: pipe_df=self.df.copy()
        assert 'question' in pipe_df.columns, "question missing in pipe_df"
        assert 'gold_items' in pipe_df.columns, "gold_items missing in pipe_df"

        fewshots_df=pipe_df[:self.n_fewshots].copy()
        pipe_df=pipe_df[self.n_fewshots:].reset_index(drop=True).copy()
        pipe_df['fewshots_df']=len(pipe_df)*[fewshots_df]

        pipe_df['prompt_messages']=pipe_df.apply(lambda x: self.get_prompt_messages(x, **kwargs), axis=1)

        # max_new_tokens=int(max([llm_pipe.count_tokens("; ".join(sample.gold_items)) for _,sample in pred_df.iterrows()])*2) #allow maximum twice the number of tokens in the gold items
        # self.logger.info(f"max_new_tokens: {max_new_tokens}")
        return pipe_df

    @staticmethod 
    def get_prompt(sample=dict(), **kwargs):
        return get_qdq_prompt(**sample, **kwargs)

    @staticmethod
    def get_eval_dict(pred_completion, gold_items, **kwargs):

        def normalize(item_str):
            item_str=item_str.strip()
            item_str=item_str.lower()
            item_str=re.sub(r'\s+', ' ', item_str)
            return item_str
        
        tp=0; tp_fuzzy=0.0; fp=0; fn=0

        for i_dont_know in ["i don't know", "i do not know", "dont know", "do not know","unsure","unknown"]:
            if fuzz.ratio(i_dont_know,pred_completion.lower())/100 > 0.9:
                return dict(tp=tp, tp_fuzzy=tp_fuzzy, fp=fp, fn=fn)

        gold_df=pd.DataFrame(gold_items, columns=['gold_item']).reset_index(names=['g'])
        gold_df['gold_val']=gold_df['gold_item'].apply(normalize)

        pred_items=list(set([c.strip() for c in re.split(r';',pred_completion)]))
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

        return dict(tp=tp, tp_fuzzy=tp_fuzzy, fp=fp, fn=fn)



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
    def get_prompt(sample=dict(),**kwargs):
        return get_ffq_prompt(**sample, **kwargs)
    