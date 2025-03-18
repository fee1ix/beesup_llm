from beesup_llm import get_labhandler, _isinstance
from beesup_llm.llm import LLMPipeline
from beesup_llm.emb import EMBPipeline

import logging
from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


from beesup_llm.injection import get_system_prompt, get_context, get_ffq_prompt


def get_ranking_plot(ranking_df: pd.DataFrame, limiters: list=[], query_txt: str=''):

    import matplotlib
    from matplotlib import pyplot as plt
    
    plt.figure(figsize=(18, 5))
    plt.plot(range(0,len(ranking_df)),ranking_df['score'])

    handles=[]
    selector_cols=[s.__class__.__name__ for s in limiters]
    for i,col in enumerate(selector_cols):
        color=matplotlib.colormaps["tab10"].colors[i+2]

        idx=len(ranking_df)-1
        val=ranking_df['score'].min()
        if False in ranking_df[col].values:
            idx=ranking_df[col].to_list().index(False)-1
            val=ranking_df.iloc[idx]['score']

        plt.axhline(y=val, color=color, linestyle='-', linewidth=1)
        plt.axvline(x=idx, color=color, linestyle='-', linewidth=1)
        handles.append(matplotlib.lines.Line2D([0], [0], color=color, lw=4, label=f'{col} ({idx}; {val:.2f})'))

    plt.legend(handles=handles)
    plt.title(f'Chunks sorted by Score\n{query_txt}'.strip())

    plt.xlabel('rank')
    plt.ylabel('score')
    plt.grid(True)
    return plt


# SELECTOR CLASSES: Used to select which first n textchunks, will be included to the briefing
# input: ranking_df .. texts sorted decreasingly by score + selection criteria
# output: list of booleans, True if text should be included, False otherwise

class RAGLimiter:
    """
    A class to represent a limiter for a Retrieval-Augmented Generation (RAG) system.

    Methods
    -------
    add_feature(chunks_df: pd.DataFrame, **kwargs):
        A static method intended to add features to a DataFrame of chunks. 
        Currently, this method is not implemented and returns nothing.

    __init__():
        Initializes an instance of the RAGLimiter class. 
        Currently, the constructor does not perform any specific initialization.
    """

    @staticmethod
    def add_feature(chunks_df: pd.DataFrame, **kwargs):
        return

    def __init__(self):

        pass
    
class NumLimiter(RAGLimiter):
    """    
    Limits the number of texts selected for further processing.
    It selects the first `n` texts based on the specified limit,
    reducing the complexity of subsequent operations.

    Attributes:
        limit (int): The maximum number of texts to select. Defaults to 100.

    Methods:
        get_mask(ranking_df: pd.DataFrame) -> list:
            Generates a boolean mask indicating which rows in the input 
            DataFrame should be selected based on the specified limit. 
            If the number of rows in the DataFrame is less than the limit, 
            all rows are selected.        
    """
    def __init__(self, limit=100, **kwargs):
        super().__init__()
        self.limit=limit

    def get_mask(self, ranking_df: pd.DataFrame) -> list:

        if self.limit>len(ranking_df):
            return [True]*len(ranking_df)

        return [True]*self.limit + [False]*(len(ranking_df)-self.limit)

class CharLimiter(RAGLimiter):
    """ 
    Selects text chunks until the cumulative number of characters reaches a specified limit.
    Attributes:
        limit (int): The maximum cumulative number of characters allowed. Defaults to 500.
    Methods:
        add_feature(chunks_df: pd.DataFrame, txt_key: str = 'text', **kwargs):
            Adds a new column 'n_chars' to the given DataFrame, which contains the 
            character count of each text chunk. If the column already exists, it is not modified.
        get_mask(ranking_df: pd.DataFrame) -> list:
            Computes a boolean mask for the input DataFrame, indicating which rows 
            can be included without exceeding the character limit. The mask is based 
            on the cumulative sum of the 'n_chars' column.
    Raises:
        AssertionError: If the 'n_chars' column is missing in the input DataFrame 
        when calling the `get_mask` method.
    """
    def __init__(self, limit=500, **kwargs):
        super().__init__()
        self.limit=limit
    
    @staticmethod
    def add_feature(chunks_df: pd.DataFrame, txt_key:str='text', **kwargs):
        if 'n_chars' not in chunks_df.columns:
            chunks_df['n_chars'] = chunks_df[txt_key].apply(len)
    
    def get_mask(self, ranking_df: pd.DataFrame) -> list:
        assert 'n_chars' in ranking_df.columns, "n_chars column missing in ranking_df"
        ranking_df['n_chars_cumsum'] = ranking_df['n_chars'].cumsum()
        return (ranking_df['n_chars_cumsum']<self.limit).to_list()

class TokenLimiter(RAGLimiter):
    """
    Limits the number of texts/chunks based on a cumulative token count.
    Attributes:
        limit (int): The maximum number of tokens allowed. Defaults to 500.
    Methods:
        add_feature(chunks_df: pd.DataFrame, txt_key: str = 'text', **kwargs):
            Adds a 'n_tokens' column to the DataFrame, representing the number of tokens
            in each text. Requires a tokenizer to be passed in kwargs.
        get_mask(ranking_df: pd.DataFrame) -> list:
            Generates a boolean mask indicating which rows in the DataFrame can be 
            included without exceeding the token limit. Requires the 'n_tokens' column 
            to be present in the DataFrame.
    """

    def __init__(self, limit=500, **kwargs):
        super().__init__()
        self.limit=limit
    
    @staticmethod
    def add_feature(chunks_df: pd.DataFrame, txt_key:str='text', **kwargs):

        assert 'tokenizer' in kwargs, "tokenizer missing in kwargs"
        tokenizer=kwargs['tokenizer']

        if 'n_tokens' not in chunks_df.columns:
            chunks_df['n_tokens'] = chunks_df[txt_key].apply(lambda x: len(tokenizer(x)['input_ids']))
    
        return 
    
    def get_mask(self, ranking_df: pd.DataFrame) -> list:
        assert 'n_tokens' in ranking_df.columns, "n_tokens column missing in ranking_df"
        ranking_df['n_tokens_cumsum'] = ranking_df['n_tokens'].cumsum()
        return (ranking_df['n_tokens_cumsum']<self.limit).to_list()

from kneed import KneeLocator
class KneeLimiter(RAGLimiter):
    """
    Selects texts/chunks with scores above the knee score using the KneeLocator algorithm.

    Attributes:
        curve (str): The type of curve to analyze. Default is "convex".
        direction (str): The direction of the curve. Default is "decreasing".

    Methods:
        __init__(curve="convex", direction="decreasing", **kwargs):
            Initializes the KneeLimiter with the specified curve and direction.

        get_mask(ranking_df: pd.DataFrame) -> list:
            Computes a mask for selecting rows in the DataFrame where the score is 
            greater than or equal to the knee score. The knee score is determined 
            using the KneeLocator algorithm.
    """

    def __init__(self, curve="convex", direction="decreasing", **kwargs):
        super().__init__()
        self.curve=curve
        self.direction=direction

    def get_mask(self, ranking_df: pd.DataFrame) -> list:
        assert 'score' in ranking_df.columns, "score column missing in ranking_df"
        knee_index = KneeLocator(
            list(range(len(ranking_df))),
            ranking_df['score'],
            curve=self.curve,
            direction=self.direction,
            ).knee
        knee_score = ranking_df.iloc[knee_index]['score']
        return (ranking_df['score']>=knee_score).to_list()

class RAGPipeline:
    """
    RAGPipeline: Retrieve and Generate Pipeline

    Args:
        ref: dict, optional
            Reference dictionary for the pipeline configuration.
        
        label: str, optional
            Label for the pipeline.
        
        chunks_df: pd.DataFrame, optional
            DataFrame containing the texts to be retrieved.

        emb_pipe: EMBPipeline, optional
            Embedding pipeline for text retrieval.
        
        llm_pipe: LLMPipeline, optional
            Language model pipeline for text generation.

        limiters: list, optional
            List of RAGLimiter instances for context limitation of the rag briefings.
        
        labh: Labhandler, optional
            Labhandler instance for reference attachment.
        
        kwargs: dict, optional
            Additional keyword arguments for pipeline configuration: txt_key, emb_key, query_txt_key, query_emb_key, chunk_instruction, query_instruction
    """

    logger=logging.getLogger(__name__)

    @classmethod
    def fit_llm_pipe(cls, llm_pipe: LLMPipeline, **kwargs) -> LLMPipeline:
        llm_pipe.generation_config['max_time'] = 1200
        llm_pipe.generation_config['max_new_tokens'] = 4096
        return llm_pipe
    
    @staticmethod
    def fit_emb_pipe(emb_pipe: EMBPipeline, **kwargs) -> EMBPipeline:
        emb_pipe.instruction = "Given a question, retrieve passages that answer the question"
        return emb_pipe

    def __init__(
            self,
            ref=None,
            label:str=None,
            chunks_df=None,
            emb_pipe=None,
            llm_pipe=None,
            limiters: list = [],
            labh=get_labhandler(),
            **kwargs):
        
        self.label = label
        self.chunk_txt_key = kwargs.get('chunk_txt_key', 'chunk')
        self.chunk_emb_key = kwargs.get('chunk_emb_key', 'emb')
        self.chunk_instruction=kwargs.get('chunk_instruction', '')

        self.query_txt_key = kwargs.get('query_txt_key', 'query')
        self.query_emb_key = kwargs.get('query_emb_key', 'emb')
        self.query_instruction=kwargs.get('query_instruction', 'Given a question, retrieve passages that answer the question')
        

        if labh is not None:
            self.labh=labh(locals())
            chunks_df=self.labh.handle_object(locals(),'chunks_df')
            llm_pipe=self.labh.handle_object(locals(),'llm_pipe')
            emb_pipe=self.labh.handle_object(locals(),'emb_pipe')
            limiters=self.labh.handle_object(locals(),'limiters')

        
        if isinstance(chunks_df, pd.DataFrame):
            chunks_df=chunks_df[[self.chunk_txt_key, self.chunk_emb_key]]
            self.chunks_df = chunks_df.copy(); del chunks_df
        
        if _isinstance(emb_pipe, EMBPipeline):
            emb_pipe = self.fit_emb_pipe(emb_pipe)
            self.emb_pipe = emb_pipe

        if _isinstance(llm_pipe, LLMPipeline):
            llm_pipe = self.fit_llm_pipe(llm_pipe)
            self.llm_pipe = llm_pipe

        if isinstance(limiters, list) and all([isinstance(l, RAGLimiter) for l in limiters]):
            self.limiters=limiters
            self.add_limiter_features()

    @property
    def df(self) -> pd.DataFrame:
        return self.chunks_df.copy()
            
    def add_limiter_features(self) -> None:

        if not hasattr(self,'llm_pipe'): return
        if not hasattr(self,'limiters'): return

        if not _isinstance(self.llm_pipe, LLMPipeline): return
        if not all(_isinstance(l, RAGLimiter) for l in self.limiters): return

        tokenizer=self.llm_pipe.get_tokenizer()
        for selector in self.limiters:
            selector.add_feature(self.chunks_df, txt_key=self.chunk_txt_key, tokenizer=tokenizer)
 
        return

    def get_query_emb(self, query_txt: str, instruction:str=None,  **kwargs) -> np.ndarray:
        query_instruction = instruction or getattr(self, 'query_instruction', None)

        return self.emb_pipe.get_emb(query_txt, instruction=query_instruction, **kwargs)

    def add_query_emb(self, pipe_df: pd.DataFrame, query_txt_key:str=None, query_emb_key:str=None, instruction:str=None, **kwargs) -> None:
        
        query_instruction = instruction or getattr(self, 'query_instruction', None)
        self.query_txt_key = query_txt_key or getattr(self, 'query_txt_key', None)
        self.query_emb_key = query_emb_key or getattr(self, 'query_emb_key', None)

        assert self.query_txt_key in pipe_df, f"{self.query_txt_key} missing in pipe_df"

        pipe_df[self.query_emb_key]=self.emb_pipe.add_emb(pipe_df, txt_key=self.query_txt_key, emb_key=self.query_emb_key, instruction=query_instruction, **kwargs)
        return

    def add_chunk_emb(self, pipe_df: pd.DataFrame, chunk_txt_key:str=None, chunk_emb_key:str=None, instruction:str=None,  **kwargs) -> np.ndarray:
        chunk_instruction = instruction or getattr(self, 'chunk_instruction', None)
        self.chunk_txt_key = chunk_txt_key or getattr(self, 'chunk_txt_key', None)
        self.chunk_emb_key = chunk_emb_key or getattr(self, 'chunk_emb_key', None)

        assert self.chunk_txt_key in pipe_df, f"{self.chunk_txt_key} missing in pipe_df"

        pipe_df[self.chunk_emb_key]=self.emb_pipe.add_emb(pipe_df, txt_key=self.chunk_txt_key, emb_key=self.chunk_emb_key, instruction=chunk_instruction, **kwargs)
        return 

    def add_ranking_df(self, pipe_df: pd.DataFrame, chunk_emb_key:str=None, query_emb_key:str=None, **kwargs) -> None:

        self.chunk_emb_key = chunk_emb_key or getattr(self, 'chunk_emb_key', None)
        self.query_emb_key = query_emb_key or getattr(self, 'query_emb_key', None)
        
        assert self.chunk_emb_key in self.chunks_df, f"{self.chunk_emb_key} missing in self.chunks_df"
        assert self.query_emb_key in pipe_df, f"{self.query_emb_key} missing in pipe_df"
    
        chunk_embs=np.vstack(self.chunks_df[self.chunk_emb_key].values)
        query_embs=np.vstack(pipe_df[self.query_emb_key].values)

        score_matrix=cosine_similarity(query_embs, chunk_embs)
        index_matrix=np.argsort(-score_matrix, axis=1)
        score_matrix=-np.sort(-score_matrix, axis=1) #sort scores descending

        ranking_dfs=[]
        for i in range(len(pipe_df)):
            ranking_df=self.chunks_df.iloc[index_matrix[i]].copy()
            ranking_df['score']=score_matrix[i]
            ranking_dfs.append(ranking_df)

        pipe_df['ranking_df']=ranking_dfs

    def get_ranking_df(self, query_emb: np.ndarray, chunk_emb_key:str=None, **kwargs) -> pd.DataFrame:
        pipe_df=pd.DataFrame({self.query_emb_key:[query_emb]})
        self.add_ranking_df(pipe_df, chunk_emb_key=chunk_emb_key, **kwargs)
        return pipe_df['ranking_df'].values[0]

    def get_briefing_df(self, ranking_df: pd.DataFrame, **kwargs) -> pd.DataFrame:

        for selector in self.limiters:
            ranking_df[selector.__class__.__name__] = selector.get_mask(ranking_df)
        
        #select only rows where all limiters are True
        briefing_df=ranking_df[ranking_df[[s.__class__.__name__ for s in self.limiters]].all(axis=1)].copy()

        return briefing_df
    
    def add_briefing_df(self, pipe_df: pd.DataFrame) -> None:
        assert 'ranking_df' in pipe_df, "ranking_df missing in pipe_df"
        pipe_df['briefing_df']=pipe_df.apply(lambda x: self.get_briefing_df(**x), axis=1)

        self.logger.debug(f"{pipe_df.iloc[0].briefing_df.columns=}")
        self._briefing_df=pipe_df.iloc[0].briefing_df

    def get_prompt_messages(self, query_txt: str, briefing_df: pd.DataFrame, **kwargs) -> list:

        kwargs['query_txt_key'] = kwargs.get('query_txt_key', self.query_txt_key)
        kwargs['chunk_txt_key'] = kwargs.get('chunk_txt_key', self.chunk_txt_key)

        prompt_messages=[]
        prompt_messages.append(
            dict(role='system', content=get_system_prompt(rag=True,**kwargs)))

        #user message with briefing texts + question
        ffq_prompt=get_ffq_prompt(query_txt, briefing_df, **kwargs)
        prompt_messages.append(dict(role='user', content=ffq_prompt))

        return prompt_messages
    
    def add_prompt_messages(self, pipe_df: pd.DataFrame, **kwargs):
        pipe_df['prompt_messages']=pipe_df.apply(lambda x: self.get_prompt_messages(**x, **kwargs), axis=1)

    def call_on_dataframe(self, pipe_df:pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.query_emb_key = kwargs.get('query_emb_key', self.query_emb_key)
        self.query_txt_key = kwargs.get('query_txt_key', self.query_txt_key)

        if self.query_emb_key not in pipe_df:
            pipe_df[self.query_emb_key]=self.add_query_emb(pipe_df, **kwargs)

        self.add_ranking_df(pipe_df, **kwargs)
        self._pipe_df=pipe_df

        self.add_briefing_df(pipe_df, **kwargs)
        self._pipe_df=pipe_df

        pipe_df.drop(columns=['ranking_df'], inplace=True)
        self.add_prompt_messages(pipe_df, **kwargs)
        self._pipe_df=pipe_df

        self.llm_pipe.add_pred_completion(pipe_df, **kwargs)
        self._pipe_df=pipe_df

        return pipe_df
    
    def call_on_single(self, query_txt:str, query_emb: np.ndarray = None, verbose=False, **kwargs) -> str:

        if query_emb is None:
            query_emb=self.get_query_emb(query_txt, **kwargs)
        
        self._query_txt=query_txt
        self._query_emb=query_emb

        ranking_df=self.get_ranking_df(query_emb, **kwargs)
        self._ranking_df=ranking_df

        briefing_df=self.get_briefing_df(ranking_df, **kwargs)
        self._briefing_df=briefing_df

        prompt_messages=self.get_prompt_messages(query_txt, briefing_df, **kwargs)
        self._prompt_messages=prompt_messages


        if verbose:
            self._plot.show()
            from IPython.display import display
            display(self._ranking)

        pred_completion = self.llm_pipe(prompt_messages,**kwargs)
        return pred_completion

    def call_on_sample(self, sample: Union[pd.Series, dict], **kwargs) -> str:
        self.query_emb_key = kwargs.get('query_emb_key', self.query_emb_key)
        self.query_txt_key = kwargs.get('query_txt_key', self.query_txt_key)

        if isinstance(sample, pd.Series): sample=sample.to_dict()

        query_txt=sample.pop(self.query_txt_key, None)
        query_emb=sample.pop(self.query_emb_key, None)

        return self.call_on_single(query_txt, query_emb, **sample, **kwargs)   

    def __call__(self, pipe_input, **kwargs):
        """
        Args: 
            pipe_input: pd.DataFrame, pd.Series, dict, str, list of str
                Input data for the pipeline

            stream: bool, optional
                If True, print the completion stream
        """
        self.add_limiter_features()
        if isinstance(pipe_input, pd.DataFrame):
            return self.call_on_dataframe(pipe_input, **kwargs)

        elif isinstance(pipe_input, (pd.Series, dict)):
            return self.call_on_sample(pipe_input, **kwargs)

        elif isinstance(pipe_input, str):
            return self.call_on_single(pipe_input, **kwargs)

        return
    

    @property
    def _plot(self):
        return get_ranking_plot(self._ranking_df, self.limiters, self._query_txt)
    @property
    def _ranking(self):
        return self._ranking_df.iloc[:len(self._briefing_df)+3] # also show some non-selected texts



