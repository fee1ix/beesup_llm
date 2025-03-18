from beesup_llm import get_labhandler, _isinstance
from beesup_llm.llm import LLMPipeline

import logging
import pandas as pd
from typing import Union


class LLMEvaluator(object):
    logger = logging.getLogger(__name__)

    @classmethod
    def fit_llm_pipe(cls, llm_pipe: LLMPipeline, **kwargs) -> LLMPipeline:
        return llm_pipe

    def __init__(
            self,
            ref=None,
            label:str=None,
            eval_df=None,
            llm_pipe=None,
            labh=get_labhandler(),
            **kwargs):
        
        self.label = label
        self.eval_epochs=kwargs.get('eval_epochs',[])

        if labh is not None:
            self.labh=labh(locals())
            eval_df=self.labh.handle_object(locals(),'eval_df', save_file=True, overwrite=False)
            llm_pipe=self.labh.handle_object(locals(),'llm_pipe')

        if isinstance(eval_df, pd.DataFrame):
            self.eval_df = eval_df.copy(); del eval_df
        
        if _isinstance(llm_pipe, LLMPipeline):
            llm_pipe = self.fit_llm_pipe(llm_pipe)
            self.llm_pipe = llm_pipe

    @property
    def df(self) -> pd.DataFrame:
        return self.eval_df.copy()

    def is_eval_epoch(self, epoch: int) -> bool:
        if self.eval_epochs==[]: return True
        elif int(epoch) in self.eval_epochs: return True
        return False

    def get_pipe_df(self, df = pd.DataFrame(), **kwargs):
        if df.empty: df=self.eval_df

        pipe_df=df.copy()
        pipe_df['prompt_messages']=pipe_df.apply(lambda x: self.get_prompt_messages(x, **kwargs), axis=1)
        return pipe_df
    
    def add_pred_completion(self, pipe_df: pd.DataFrame, llm_pipe:LLMPipeline=None, **kwargs) -> None:

        if _isinstance(llm_pipe, LLMPipeline):
            llm_pipe = self.fit_llm_pipe(llm_pipe)
            self.logger.debug(f"using llm_pipe from arg")
        else:
            llm_pipe=self.llm_pipe
            self.logger.debug(f"using llm_pipe from self")

        pipe_df=llm_pipe.add_pred_completion(pipe_df, **kwargs)

    @staticmethod
    def get_eval_dict(**kwargs) -> dict:
        return dict(info="NotImplemented")
    
    def add_eval_dict(self, pipe_df: pd.DataFrame, **kwargs) -> None:
        assert 'pred_completion' in pipe_df.columns, "pred_completion missing in pipe_df"
        pipe_df['eval_dict']=pipe_df.apply(lambda x: self.get_eval_dict(**x, **kwargs), axis=1)



    def __call__(self, llm_pipe:LLMPipeline=None, pipe_df:pd.DataFrame=None, **kwargs) -> pd.DataFrame:

        if pipe_df is None:
            pipe_df=self.get_pipe_df(**kwargs)

        self.add_pred_completion(pipe_df, llm_pipe=llm_pipe, **kwargs)
        self.add_eval_dict(pipe_df, **kwargs)
                
        return pipe_df