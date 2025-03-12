# from beesup_llm import *

# from ..toolkit.setup_utils import *
# from ..toolkit.llm_utils import *

# from .extraction_utils import *

# from beesup_llm.dataset import *
# #from beesup_llm.model import *
# from beesup_llm.model_pipelines import *

# from datasets import Dataset

from beesup_llm import get_labhandler, _isinstance
from beesup_llm.extraction.extraction_utils import *
from beesup_llm.extraction.evaluation_utils import *
from beesup_llm.llm import LLMPipeline

import logging
from typing import Union

import pandas as pd


class ExtractionSample(object):
    
    def __init__(self, pred_completion:str, gold_completion:str=None, **kwargs) -> None:

        self.pred_completion=pred_completion
        self.gold_completion=gold_completion

        # for k,val in kwargs.items(): 
        #     setattr(self,k,val)

        self.parse_json(prefix='pred', exclude_none=True)
        self.parse_df(prefix='pred')

        if gold_completion is not None:
            self.parse_json(prefix='gold', exclude_none=True)
            self.parse_df(prefix='gold')
            self.evaluate()

    def parse_json(self, prefix='pred', exclude_none=True):
        completion=getattr(self, f'{prefix}_completion')
        the_json, is_valid, is_empty = pydantic_parse(completion, exclude_none)

        if is_valid==False:
            the_json={k:None for k in ExtractionScheme4MultipeObservations.model_fields.keys()}

        setattr(self, f'{prefix}_json', the_json)
        setattr(self, f'{prefix}_is_valid', is_valid)
        setattr(self, f'{prefix}_is_empty', is_empty)
        
        return 
    
    def parse_df(self, prefix='pred', create_meta_row=False):
        """
        create_meta_row=False: meta attributes are backpropagated to the individual observations, this is the default for evaluation!!
        create_meta_row=True is only used for generating the highlightning!
        """

        the_json=getattr(self, f'{prefix}_json')
        the_df=tabelize_json(the_json,create_meta_row=create_meta_row) # create_meta_row=False means that meta attributes are backpropagated to the individual observations

        the_df.attrs['is_empty']=getattr(self, f'{prefix}_is_empty')
        the_df.attrs['is_valid']=getattr(self, f'{prefix}_is_valid') 
        
        if create_meta_row==True: setattr(self,f'raw_{prefix}_df',the_df)
        else: setattr(self,f'{prefix}_df',the_df)

        return 
    
    def evaluate(self):

        gold_df=self.gold_df
        pred_df=self.pred_df
        
        match_df=get_match(gold_df,pred_df)
        self.match_df=match_df

        errors_df=get_errors(match_df,gold_df,pred_df)
        self.errors_df=errors_df

        conf_dict=get_conf_dict(errors_df)
        self.conf_dict=conf_dict

        eval_dict=get_eval_dict(conf_dict)
        self.eval_dict=eval_dict

        self.total_score=eval_dict['total_score']
    
    def evaluate(self):

        gold_df=self.gold_df
        pred_df=self.pred_df
        
        match_df=get_match(gold_df,pred_df)
        self.match_df=match_df

        errors_df=get_errors(match_df,gold_df,pred_df)
        self.errors_df=errors_df

        conf_dict=get_conf_dict(errors_df)
        self.conf_dict=conf_dict

        eval_dict=get_eval_dict(conf_dict)
        self.eval_dict=eval_dict

        self.total_score=eval_dict['total_score']
    
    def evaluate_raw(self):

        self.parse_df(prefix='gold', create_meta_row=True)
        self.parse_df(prefix='pred', create_meta_row=True)

        raw_gold_df=self.raw_gold_df
        raw_pred_df=self.raw_pred_df

        raw_match_df=get_match(raw_gold_df,raw_pred_df)
        self.raw_match_df=raw_match_df

        raw_errors_df=get_errors(raw_match_df,raw_gold_df,raw_pred_df)
        self.raw_errors_df=raw_errors_df

        raw_conf_dict=get_conf_dict(raw_errors_df)
        self.raw_conf_dict=raw_conf_dict

        raw_eval_dict=get_eval_dict(raw_conf_dict)
        self.raw_eval_dict=raw_eval_dict

        self.raw_total_score=raw_eval_dict['total_score']

    def load_highlighting(self):

        self.evaluate_raw()
        self.raw_errors_df=get_error_spans(self.raw_errors_df,self.gold_completion,self.pred_completion,verbose=False)
        self.gold_highlighting=get_char_highlighting(self.gold_completion,self.raw_errors_df,col='gold')
        self.pred_highlighting=get_char_highlighting(self.pred_completion,self.raw_errors_df,col='pred')
        
    def get_errors_df(self):
        errors_df=self.errors_df.copy()
        return errors_df

    def __repr__(self):
        if self.gold_completion is not None:
            if not hasattr(self, 'gold_highlighting'): self.load_highlighting()
            if not hasattr(self, 'pred_highlighting'): self.load_highlighting()

            from beede_llm.src.utils import print_multicol
            print_multicol(["<h2>GOLD-LABEL</h2>",f"<h2>PREDICTION (S<sub>extract</sub>={self.total_score:.3f})</h2>"])
            print_multicol([self.gold_highlighting,self.pred_highlighting])
        return ''

class ExtractionPipeline(object):

    logger=logging.getLogger(__name__)

    @classmethod
    def fit_llm_pipe(cls, llm_pipe: LLMPipeline, **kwargs) -> LLMPipeline:
        llm_pipe.generation_config['stop_strings'] = ['}\n```']
        return llm_pipe

    def __init__(
            self,
            ref=None,
            label:str=None,
            llm_pipe=None,
            labh=get_labhandler(),
            **kwargs):
        
        self.label = label
        self.use_extraction_prompt = kwargs.get('use_extraction_prompt', True)
        self.use_few_shots = kwargs.get('use_few_shots', True)

        if labh is not None:
            self.labh=labh(locals())
            llm_pipe=self.labh.handle_object(locals(),'llm_pipe')


        if _isinstance(llm_pipe, LLMPipeline):
            llm_pipe = self.fit_llm_pipe(llm_pipe)
            self.llm_pipe = llm_pipe

    def get_prompt_messages(self, report_passage: str, **kwargs) -> list:

        kwargs['use_few_shots'] = kwargs.get('use_few_shots', self.use_few_shots)
        kwargs['use_extraction_prompt'] = kwargs.get('use_extraction_prompt', self.use_extraction_prompt)

        return get_prompt_messages(report_passage, **kwargs)

    def add_prompt_messages(self, pipe_df: pd.DataFrame, **kwargs):
        pipe_df['prompt_messages']=pipe_df.apply(lambda x: self.get_prompt_messages(**x, **kwargs), axis=1)

    def get_pred_dict(self, pred_completion: str, **kwargs) -> dict:
        pred_dict=dict()

        extraction_sample=ExtractionSample(pred_completion)
        pred_dict['pred_json']=extraction_sample.pred_json
        pred_dict['pred_is_valid']=extraction_sample.pred_is_valid
        pred_dict['pred_is_empty']=extraction_sample.pred_is_empty

        self._pred=extraction_sample
        self._pred_dict=pred_dict

        return pred_dict
    
    def add_pred_dict(self, pipe_df: pd.DataFrame, **kwargs) -> None:
        assert 'pred_completion' in pipe_df.columns, "missing 'pred_completion' column"
        pipe_df['pred_dict']=pipe_df['pred_completion'].apply(lambda x: self.get_pred_dict(x, **kwargs))
        #pipe_df = pipe_df.join(pd.json_normalize(pipe_df['pred_dict']))
        return
    
    def get_eval_dict(self, pred_completion:str, gold_completion:str, **kwargs) -> dict:
        evaluation_sample=ExtractionSample(pred_completion, gold_completion)

        self._eval=evaluation_sample
        self._eval_dict=evaluation_sample.eval_dict

        return evaluation_sample.eval_dict
    
    def add_eval_dict(self, pipe_df: pd.DataFrame, **kwargs) -> None:
        assert 'pred_completion' in pipe_df.columns, "missing 'pred_completion' column"
        assert 'gold_completion' in pipe_df.columns, "missing 'gold_completion' column"
        pipe_df['eval_dict']=pipe_df.apply(lambda x: self.get_eval_dict(**x), axis=1)
        #pipe_df = pipe_df.join(pd.json_normalize(pipe_df['eval_dict']))
        return

    def call_on_dataframe(self, pipe_df:pd.DataFrame, **kwargs) -> pd.DataFrame:
        llm_pipe=self.fit_llm_pipe(kwargs.get('llm_pipe', self.llm_pipe))

        self.add_prompt_messages(pipe_df, **kwargs)
        llm_pipe.add_pred_completion(pipe_df, **kwargs)
        self.add_pred_dict(pipe_df)

        if 'gold_completion' in pipe_df.columns:
            self.add_eval_dict(pipe_df, **kwargs)

        return pipe_df

    def call_on_single(self, report_passage:str, gold_completion:str=None,  **kwargs) -> str:
        llm_pipe=self.fit_llm_pipe(kwargs.get('llm_pipe', self.llm_pipe))

        prompt_messages=get_prompt_messages(report_passage, **kwargs)
        pred_completion = llm_pipe(prompt_messages, **kwargs) #streaming possible
        pred_dict=self.get_pred_dict(pred_completion, **kwargs)
        self._pred_dict=pred_dict

        if gold_completion is not None:
            eval_dict=self.get_eval_dict(pred_completion, gold_completion, **kwargs)
            self._eval_dict=eval_dict

        return pred_dict['pred_json']

    def call_on_sample(self, sample:Union[pd.Series, dict], **kwargs) -> str:

        if isinstance(sample, pd.Series): sample=sample.to_dict()
        report_passage=sample.pop('report_passage')
        return self.call_on_single(report_passage, **sample, **kwargs)   

    def __call__(self, pipe_input, **kwargs):

        if isinstance(pipe_input, pd.DataFrame):
            return self.call_on_dataframe(pipe_input, **kwargs)
        
        elif isinstance(pipe_input, (pd.Series, dict)):
            return self.call_on_sample(pipe_input, **kwargs)
        
        else:
            return self.call_on_single(pipe_input, **kwargs)

    

    # @staticmethod
    # def get_pred_parse_only(pred_completion, **kwargs):
    #     sample=ExtractionSample(pred_completion=pred_completion)
    #     sample.parse_df()
    #     return sample.pred_df

    # def get_pred(self, report_passage, llm_pipe, **kwargs):

    #     llm_pipe.prepare_inference()
    #     prompt_messages=get_prompt_messages(report_passage, **self.get_prompting_config(**kwargs))

    #     pred_completion=''
    #     for new_token in llm_pipe.get_pipeline_stream(prompt_messages, **kwargs):
    #         pred_completion+=new_token
    #         print(new_token, end='', flush=True)
        
    #     return self.get_pred_parse_only(pred_completion,**kwargs)

    # def prepare_df_for_completion(self, df, **kwargs):
    #     assert 'report_passage' in df.columns, "df must have 'prompt_messages' column"
    #     df['prompt_messages']=df['report_passage'].apply(lambda x: get_prompt_messages(x,**self.get_prompting_config(**kwargs)))
    #     return df
    
    # def prepare_df_for_finetuning(self, df, **kwargs):
    #     assert 'gold_completion' in df.columns, "df must have 'gold_completion' column"
    #     df=self.prepare_df_for_completion(df, **kwargs)
    #     df['gold_message']=df['gold_completion'].apply(lambda x: [{'role':'assistant','content': x}])
    #     return df

    # def get_ds_for_finetuning(self, df, tokenizer, **kwargs):
    #     assert 'prompt_messages' in df.columns, "df must have 'prompt_messages' column"
    #     assert 'gold_message' in df.columns, "df must have 'gold_message' column"
    #     ds=Dataset.from_list(df.apply(lambda x: prepare_sample_for_chat_finetuning(x, tokenizer),axis=1).to_list())
    #     return ds
    
    # def get_ds_for_completion(self, df, tokenizer, **kwargs):
    #     assert 'prompt_messages' in df.columns, "df must have 'prompt_messages' column"
    #     ds=Dataset.from_list(df.apply(lambda x: prepare_sample_for_chat_completion(x, tokenizer),axis=1).to_list())
    #     return ds

    # @staticmethod
    # def get_pred_df_parse_only(df, **kwargs):
    #     assert 'pred_completion' in df.columns, "missing 'pred_completion' column"
        
    #     df[['pred_json','pred_is_valid','pred_is_empty']]=None,None,None
    #     for i,row in df.iterrows():
    #         try:
    #             sample=ExtractionSample(pred_completion=row['pred_completion'])
    #             df.at[i,'pred_json']=sample.pred_json
    #             df.at[i,'pred_is_valid']=sample.pred_is_valid
    #             df.at[i,'pred_is_empty']=sample.pred_is_empty

    #         except: pass
        
    #     return df

    # def get_pred_df(self, df, llm_pipe, **kwargs):

    #     df = self.prepare_df_for_completion(df, **kwargs)
    #     llm_pipe.prepare_inference()

    #     df = llm_pipe.get_pred_df(df, **kwargs)
    #     self.llm_pipe._recent_generation_config=llm_pipe._recent_generation_config
    #     df = self.get_pred_df_parse_only(df)
    #     return df
 
    # def __call__(self, the_input, llm_ref=None, **kwargs):

    #     if llm_ref:
    #         llm_pipe=LanguageModelPipeline.from_ref(llm_ref)
    #         llm_pipe.update_config(self.llm_config)
    #         #llm_pipe.update_config(self._default_config['llm_config'])
    
    #     elif hasattr(self, 'llm_pipe'):
    #         llm_pipe=self.llm_pipe

    #     if isinstance(the_input, str): #input is a single report passage
    #         return self.get_pred(the_input, llm_pipe, **kwargs)
        
    #     elif isinstance(the_input, pd.DataFrame): #input is a dataframe containing a column 'report_passage'
    #         return self.get_pred_df(the_input, llm_pipe, **kwargs)

    #     elif isinstance(the_input, Dataset):
    #         self.logger.info("Dataset input detected")

    #     return



    

    






        

 
 


    
    



