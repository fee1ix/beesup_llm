
from beesup_llm import get_labhandler, _isinstance
from beesup_llm.llm_evaluation import *
from beesup_llm.injection import *
#from beesup_llm.injection.taxomizer import *
#from beesup_llm.injection.rag import RAGPipeline
#from beesup_llm.injection.injection_experiment import InjectionExperiment

import re
from typing import Union
from rapidfuzz import fuzz
from transformers import TrainerCallback

# class EvaluatorCallback(TrainerCallback):

#     def __init__(
#             self,
#             evaluator,
#             experiment: InjectionExperiment,
#             rag_pipe: RAGPipeline = None
#             ):
        
#         self.evaluator=evaluator
#         self.experiment=experiment
#         self.rag_pipe=rag_pipe

#         self.name=f"{evaluator.subtype}:{evaluator.id}"

#         if self.rag_pipe:
#             self.name+=f"-rag:{rag_pipe.id}"
            
#     def save_callback_df(self, callback_df:pd.DataFrame, epoch=0, global_step=0, **kwargs):

#         epoch, global_step = int(epoch), int(global_step)

#         save_path=f"{self.experiment._path}/{epoch}:{global_step}_{self.name}_callback_df.pkl"
#         callback_df.to_pickle(save_path)
#         self.experiment.logger.info(f"Saved {self.name} to {save_path}")
    
#     def on_epoch_end(self, args=None, state=None, control=None, **kwargs):

#         if not self.evaluator.is_eval_epoch(state.epoch):
#             self.experiment.logger.info(f"{self.name}\tepoch: {state.epoch}\tglobal step: {state.global_step} not an eval epoch")
#             return #skip evaluation if not specified as eval epoch
        
#         toc=None
#         if hasattr(self.experiment,'taxomizer'):
#             toc=self.experiment.taxomizer.get_table_of_contents()

#         self.experiment.logger.info(f"epoch: {state.epoch}\tglobal step: {state.global_step}")

#         model=kwargs['model']
#         model.eval()

#         if self.rag_pipe:
#             self.rag_pipe.add_ranking_df(self.evaluator.df, **kwargs)
#             self.rag_pipe.add_briefing_df(self.evaluator.df, **kwargs)
            

#         callback_df=self.evaluator(llm_ref=model, toc=toc, **kwargs)
#         self.save_callback_df(callback_df, **state.__dict__)
#         self.evaluator.load_df() #reload evaluator df to remove ranking and briefing columns

# class MCEEvaluatorCallback(EvaluatorCallback):
#     """Multiclass Cross Entropy Loss Evaluator Callback

#     fetches sample-mapped loss data from Custom Trainer Wrapper
#     """

#     def __init__(self, experiment):
#         self.experiment=experiment
#         self.name=f"mce_callback"
#         self.loss_data=[]
    
#     def add_loss_data(self, data):
#         self.loss_data.extend(data)

#     def on_epoch_end(self, args, state, control, **kwargs):

#         callback_df=pd.DataFrame(self.loss_data)

#         self.loss_data = []
#         self.save_df(callback_df, state)


class InjectionEvaluator(LLMEvaluator):
    def get_prompt_messages(self, sample: Union[pd.Series, dict], **kwargs) -> list:

        prompt_messages=[]
        prompt_messages.append(dict(role='system', content=get_system_prompt(**sample, **kwargs)))
        prompt_messages.append(dict(role='user', content=self.get_prompt(**sample, **kwargs)))
        return prompt_messages
        
class MCQEvaluator(InjectionEvaluator):
    """Multiple Choice Question Evaluator"""

    @classmethod
    def fit_llm_pipe(cls, llm_pipe: LLMPipeline, **kwargs) -> LLMPipeline:
        llm_pipe=super().fit_llm_pipe(llm_pipe, **kwargs)
        llm_pipe.generation_config['stop_strings'] = ['\n']
        llm_pipe.generation_config['max_new_tokens'] = 8
        return llm_pipe

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
    def get_prompt(sample:dict=dict(), **kwargs) -> str:
        return get_mcq_prompt(**sample, **kwargs)

    @staticmethod
    def get_eval_dict(pred_completion:str, answer:int, **kwargs) -> dict:

        gold_choice=chr(65+int(answer)).lower()

        pred_choice=pred_completion.strip().lower()
        pred_choice=re.sub(r'[^a-zA-Z]','',pred_choice)
        #TODO: log if pred_choice really has len==1, if not it could be something else than the correct letter
        pred_choice=pred_choice[:1]
        
        eval_dict=dict(
            pred_choice	= pred_choice,
            gold_choice	= gold_choice,
            tp = 1 if gold_choice==pred_choice else 0
        )

        return eval_dict
    
class QDQEvaluator(InjectionEvaluator):
    """Query Driven Questions Evaluator"""

    @classmethod
    def fit_llm_pipe(cls, llm_pipe: LLMPipeline, **kwargs) -> LLMPipeline:
        llm_pipe=super().fit_llm_pipe(llm_pipe, **kwargs)
        llm_pipe.generation_config['stop_strings'] = ['\n']
        return llm_pipe

    def __init__(self, ref=None, n_fewshots=3, **kwargs):
        super().__init__(ref, **kwargs)
        self.n_fewshots=n_fewshots

    def get_pipe_df(self, pipe_df = pd.DataFrame(), **kwargs) -> pd.DataFrame:
        if pipe_df.empty: pipe_df=self.eval_df.copy()
        assert 'question' in pipe_df.columns, "question missing in pipe_df"
        assert 'gold_items' in pipe_df.columns, "gold_items missing in pipe_df"

        fewshots_df=pipe_df[:self.n_fewshots].copy()
        pipe_df=pipe_df[self.n_fewshots:].reset_index(drop=True).copy()
        pipe_df['fewshots_df']=len(pipe_df)*[fewshots_df]

        pipe_df['prompt_messages']=pipe_df.apply(lambda x: self.get_prompt_messages(x, **kwargs), axis=1)

        return pipe_df

    @staticmethod 
    def get_prompt(sample:Union[pd.Series,dict]=dict(), **kwargs) -> str:
        return get_qdq_prompt(**sample, **kwargs)

    @staticmethod
    def get_eval_dict(pred_completion: str, gold_items:list, **kwargs) -> dict:

        def normalize(item_str:str) -> str:
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

class FFQEvaluator(InjectionEvaluator):
    """Free Form Question Evaluator"""

    @classmethod
    def fit_llm_pipe(cls, llm_pipe: LLMPipeline, **kwargs) -> LLMPipeline:
        llm_pipe=super().fit_llm_pipe(llm_pipe, **kwargs)
        llm_pipe.generation_config['stop_strings'] = ['\n']
        llm_pipe.generation_config['max_new_tokens'] = 500
        return llm_pipe

    @staticmethod 
    def get_prompt(sample:Union[pd.Series, dict]=dict(),**kwargs) -> str:
        return get_ffq_prompt(**sample, **kwargs)
    