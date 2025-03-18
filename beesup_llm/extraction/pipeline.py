from beesup_llm import get_labhandler, _isinstance
from beesup_llm.extraction.utils import *
from beesup_llm.extraction.evaluation_utils import *
from beesup_llm.llm import LLMPipeline

import logging
from typing import Union

import pandas as pd

class ExtractionSample:
    """   
    ExtractionSample is a class designed to evaluate and compare predicted and gold-standard completions 
    in the context of data extraction tasks. It provides methods for parsing, evaluating, and highlighting 
    differences between the predicted and gold-standard data.
    Attributes:
        pred_completion (str): The predicted completion string.
        gold_completion (str): The gold-standard completion string. Default is None.
        match_df (DataFrame): DataFrame containing matches between gold and predicted data.
        errors_df (DataFrame): DataFrame containing errors between gold and predicted data.
        conf_dict (dict): Dictionary containing confidence metrics.
        eval_dict (dict): Dictionary containing evaluation metrics.
        total_score (float): Total evaluation score.
        raw_match_df (DataFrame): DataFrame containing raw matches for highlighting.
        raw_errors_df (DataFrame): DataFrame containing raw errors for highlighting.
        raw_conf_dict (dict): Dictionary containing raw confidence metrics.
        raw_eval_dict (dict): Dictionary containing raw evaluation metrics.
        raw_total_score (float): Total raw evaluation score.
        gold_highlighting (str): Highlighted gold completion string.
        pred_highlighting (str): Highlighted predicted completion string.
    Methods:
        __init__(pred_completion: str, gold_completion: str = None, **kwargs):
            Initializes the ExtractionSample object, parses the input data, and evaluates if gold_completion is provided.
        parse_json(prefix='pred', exclude_none=True):
            Parses the completion string into a JSON object and validates it.
        parse_df(prefix='pred', create_meta_row=False):
            Converts the parsed JSON object into a DataFrame for further processing.
        evaluate():
            Compares the gold and predicted DataFrames, calculates matches, errors, and evaluation metrics.
        evaluate_raw():
            Performs raw evaluation by creating meta rows in the DataFrames for highlighting purposes.
        load_highlighting():
            Generates character-level highlighting for gold and predicted completions based on errors.
        get_errors_df():
            Returns a copy of the errors DataFrame.
        __repr__():
            Provides a formatted representation of the object, including highlighted gold and predicted completions.
        """
    
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

            from beesup_llm.toolkit.visualization import print_multicol
            print_multicol(["<h2>GOLD-LABEL</h2>",f"<h2>PREDICTION (S<sub>extract</sub>={self.total_score:.3f})</h2>"])
            print_multicol([self.gold_highlighting,self.pred_highlighting])
        return ''

class ExtractionPipeline:

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

    def call_on_single(self, report_passage:str, gold_completion:str=None, **kwargs) -> str:
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

    
    






        

 
 


    
    



