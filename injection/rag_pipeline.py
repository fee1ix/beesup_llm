
from beesup_llm import get_labhandler
from beesup_llm.llm import *
#from beesup_llm.model_pipelines import *
#from beesup_llm.dataset import *

import pandas as pd
from typing import Union
from beesup_llm.injection import get_system_prompt, get_context, get_ffq_prompt

#from beesup_llm.injection.evaluator import Evaluator

def plot_ranking(ranking_df: pd.DataFrame, selectors: list=[], question: str=''):

    import matplotlib
    from matplotlib import pyplot as plt
    
    plt.figure(figsize=(18, 5))
    plt.plot(range(0,len(ranking_df)),ranking_df['score'])

    handles=[]
    selector_cols=[s.__class__.__name__ for s in selectors]
    for i,col in enumerate(selector_cols):
        color=matplotlib.colormaps["tab10"].colors[i+2]
        idx=ranking_df[col].to_list().index(False)-1
        val=ranking_df.iloc[idx]['score']

        plt.axhline(y=val, color=color, linestyle='-', linewidth=1)
        plt.axvline(x=idx, color=color, linestyle='-', linewidth=1)
        #plt.text(idx, ranking_df['score'].min(), f"({idx};{val:.2f})", color=color, fontsize=10, rotation=90, va='bottom', ha='right')
        handles.append(matplotlib.lines.Line2D([0], [0], color=color, lw=4, label=f'{col} ({idx}; {val:.2f})'))

    plt.legend(handles=handles)
    plt.title(f'texts sorted by score\n{question}'.strip())

    plt.xlabel('rank')
    plt.ylabel('score')
    plt.grid(True)
    plt.show()


# SELECTOR CLASSES: Used to select which first n texts will be included to the briefing
# input: ranking_df .. texts sorted decreasingly by score + selection criteria
# output: list of booleans, True if text should be included, False otherwise

class RAGSelector(object):
    type='rag_selector'

    @classmethod
    def from_ref(cls, ref=dict(), **kwargs):

        if isinstance(ref, cls): return ref
        if isinstance(ref, dict):
            if PreTrimLimit.matches(**ref): return PreTrimLimit(**ref,**kwargs)
            if FitTokenLimit.matches(**ref): return FitTokenLimit(**ref,**kwargs)
            if FitCharLimit.matches(**ref): return FitCharLimit(**ref,**kwargs)
            if FitKneeScore.matches(**ref): return FitKneeScore(**ref,**kwargs)

        return cls(**kwargs)

    @classmethod
    def matches(cls, name=None, **kwargs):
        if name == cls.__name__: return True
        return False

    def __init__(self):
        self.name=self.__class__.__name__
    
class PreTrimLimit(RAGSelector):
    """
    Selects the first n texts, reduces complexity for following operations
    """
    def __init__(self, limit=100, **kwargs):
        super().__init__()
        self.limit=limit

    @staticmethod
    def add_feature(chunks_df: pd.DataFrame, **kwargs):
        return
    
    def get_mask(self, ranking_df: pd.DataFrame) -> list:
        return [True]*self.limit + [False]*(len(ranking_df)-self.limit)

class FitTokenLimit(RAGSelector):
    """
    Selects texts until the cumulative number of tokens reaches the limit
    """
    def __init__(self, limit=500, **kwargs):
        super().__init__()
        self.limit=limit
    
    @staticmethod
    def add_feature(chunks_df: pd.DataFrame, **kwargs):

        assert 'tokenizer' in kwargs, "tokenizer missing in kwargs"
        tokenizer=kwargs['tokenizer']

        if 'n_tokens' not in chunks_df.columns:
            chunks_df['n_tokens'] = chunks_df['text'].apply(lambda x: len(tokenizer(x)['input_ids']))
    
        return 
    
    def get_mask(self, ranking_df: pd.DataFrame) -> list:
        assert 'n_tokens' in ranking_df.columns, "n_tokens column missing in ranking_df"
        ranking_df['n_tokens_cumsum'] = ranking_df['n_tokens'].cumsum()
        return (ranking_df['n_tokens_cumsum']<self.limit).to_list()

class FitCharLimit(RAGSelector):
    """
    Selects texts until the cumulative number of characters reaches the limit
    """
    def __init__(self, limit=500, **kwargs):
        super().__init__()
        self.limit=limit
    
    @staticmethod
    def add_feature(chunks_df: pd.DataFrame, **kwargs):
        if 'n_chars' not in chunks_df.columns:
            chunks_df['n_chars'] = chunks_df['text'].apply(len)
    
    def get_mask(self, ranking_df: pd.DataFrame) -> list:
        assert 'n_chars' in ranking_df.columns, "n_chars column missing in ranking_df"
        ranking_df['n_chars_cumsum'] = ranking_df['n_chars'].cumsum()
        return (ranking_df['n_chars_cumsum']<self.limit).to_list()

from kneed import KneeLocator
class FitKneeScore(RAGSelector):
    """
    Selects texts with scores above the knee score
    """
    def __init__(self, curve="convex", direction="decreasing", **kwargs):
        super().__init__()
        self.curve=curve
        self.direction=direction

    @staticmethod
    def add_feature(chunks_df: pd.DataFrame, **kwargs):
        return
    
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


from sklearn.metrics.pairwise import cosine_similarity

class RAGPipeline(object):

    @staticmethod
    def fit_llm_pipe(llm_pipe: LLMPipeline, **kwargs) -> LLMPipeline:
        llm_pipe.generation_config['max_time'] = 1200
        llm_pipe.generation_config['max_new_tokens'] = 4096
        return llm_pipe

    def __init__(
            self,
            ref=None,
            label:str=None,
            data=None,
            llm_pipe=None,
            selectors: list = [],
            labh=get_labhandler(),
            **kwargs):
        
        self.emb_col = kwargs.get('emb_col', 'spo')
        self.text_col = kwargs.get('text_col', 'spo')

        if labh is not None:
            self.labh=labh
            llm_pipe, data, selectors = self.labh.attach(locals(), var_names=['llm_pipe', 'data', 'selectors'])
        
        if isinstance(llm_pipe, LLMPipeline):
            llm_pipe = self.fit_llm_pipe(llm_pipe)
            self.llm_pipe = llm_pipe
    
        if isinstance(data, pd.DataFrame):
            df=data.copy()
            df.rename(columns={f"{self.emb_col}_emb":"emb", f"{self.text_col}":"text"}, inplace=True)
            df=df[['text','emb']]
            self.df = df
            self.add_selector_features()

        if selectors:
            self.selectors = [RAGSelector.from_ref(s) for s in selectors]

        # if dataset_ref:
        #     dataset=BaseDataset.from_ref(dataset_ref)
        #     self.dataset_config=self.dataset.get_config()
        #     dataset_df=dataset.df

        #     if not dataset.parent_config: self.logger.warning('Dataset has no parent config')
        #     parent_df=BaseDataset(dataset.parent_config).df

        #     # add embedding combinations
        #     add_cols=[self.text_col,'source_name','attr_type','n_units','n_words']
        #     dataset_df=dataset_df.merge(parent_df[add_cols+[f"{self.emb_col}_emb"]], left_on='kidx', right_index=True, how='left')
        #     dataset_df.rename(columns={f"{self.emb_col}_emb":'emb', f"{self.text_col}":'text'}, inplace=True)
        #     self.df=dataset_df
            
   
        # if llm_ref:
        #     self.llm_pipe=LanguageModelPipeline.from_ref(llm_ref)
        #     self.llm_pipe.update_config(self.llm_config)
        #     self.llm_pipe.update_config_smart(kwargs)

        
        # self.selectors=[]
        # if selector_refs:
        #     self.selectors=[RAGSelector.from_ref(ref) for ref in selector_refs]
        #     self.selector_configs=[s.__dict__ for s in self.selectors]

        # self.add_selector_features()

    def add_selector_features(self):

        tokenizer=self.llm_pipe.get_tokenizer()
        for selector in self.selectors:
            selector.add_feature(self.df, tokenizer=tokenizer)
 
        return

    def add_ranking_df(self, pipe_df: pd.DataFrame) -> None:
        assert 'question_emb' in pipe_df, "question_emb missing in samples"

        text_embs=np.vstack(self.df['emb'].values)
        question_embs=np.vstack(pipe_df['question_emb'].values)

        score_matrix=cosine_similarity(question_embs,text_embs)
        index_matrix=np.argsort(-score_matrix, axis=1)
        score_matrix=-np.sort(-score_matrix, axis=1) #sort scores descending

        ranking_dfs=[]
        for i in range(len(pipe_df)):
            ranking_df=self.df.iloc[index_matrix[i]].copy()
            ranking_df['score']=score_matrix[i]
            ranking_df.drop(columns=['prompt_messages','gold_message','split','source_name','attr_type'], errors='ignore', inplace=True)
            ranking_dfs.append(ranking_df)

        pipe_df['ranking_df']=ranking_dfs

    def get_ranking_df(self, question_emb: np.ndarray, **kwargs) -> pd.DataFrame:

        question_df=pd.DataFrame({'question_emb':[question_emb]})
        self.add_ranking_df(question_df)
        #display(question_df)
        return question_df['ranking_df'].values[0]

        # text_embs=np.vstack(self.df['emb'].values)
        # question_emb=sample['question_emb']

        # ranking_df=self.df.copy()
        # ranking_df['score']=cosine_similarity(question_emb.reshape(1,-1),text_embs)[0]
        # ranking_df=ranking_df.sort_values('score', ascending=False)

        # return ranking_df

    def get_briefing_df(self, ranking_df: pd.DataFrame, **kwargs) -> pd.DataFrame:

        for selector in self.selectors:
            ranking_df[selector.__class__.__name__] = selector.get_mask(ranking_df)
        
        #select only rows where all selectors are True
        briefing_df=ranking_df[ranking_df[[s.__class__.__name__ for s in self.selectors]].all(axis=1)].copy()

        return briefing_df
    
    def add_briefing_df(self, pipe_df: pd.DataFrame) -> None:
        assert 'ranking_df' in pipe_df, "ranking_df missing in pipe_df"
        pipe_df['briefing_df']=pipe_df.apply(lambda x: self.get_briefing_df(**x), axis=1)

    def get_prompt_messages(self, question: str, briefing_df: pd.DataFrame, **kwargs) -> list:

        prompt_messages=[]
        prompt_messages.append(
            dict(role='system', content=get_system_prompt(rag=True,**kwargs)))

        #user message with briefing texts + question
        ffq_prompt=get_ffq_prompt(question, briefing_df)
        prompt_messages.append(dict(role='user', content=ffq_prompt))

        return prompt_messages
    
    def add_prompt_messages(self, pipe_df: pd.DataFrame, **kwargs):
        pipe_df['prompt_messages']=pipe_df.apply(lambda x: self.get_prompt_messages(**x, **kwargs), axis=1)

    def call_on_dataframe(self, pipe_df:pd.DataFrame, **kwargs) -> pd.DataFrame:

        self.add_ranking_df(pipe_df)
        self.add_briefing_df(pipe_df)
        pipe_df.drop(columns=['ranking_df'], inplace=True)
        self.add_prompt_messages(pipe_df, **kwargs)

        self.llm_pipe.add_pred_completion(pipe_df, **kwargs)
        return pipe_df
    
    def call_on_single(self, question: str, question_emb: np.ndarray = None, verbose=False, **kwargs) -> str:

        if question_emb is None:
            raise NotImplementedError("question_emb is required for RAGPipeline")

        ranking_df=self.get_ranking_df(question_emb)
        briefing_df=self.get_briefing_df(ranking_df)
        prompt_messages=self.get_prompt_messages(question, briefing_df)

        self._ranking_df=ranking_df
        self._briefing_df=briefing_df
        self._prompt_messages=prompt_messages

        if verbose:
            plot_ranking(ranking_df, self.selectors, question)
            from IPython.display import display
            display(ranking_df.iloc[0:len(briefing_df)+3]) # also show some non-selected texts
        
        pred_completion = self.llm_pipe(prompt_messages,**kwargs)

        return pred_completion

    def call_on_sample(self, sample: Union[pd.Series, dict], **kwargs) -> str:
        return self.call_on_single(**sample, **kwargs)

    def __call__(self, pipe_input, **kwargs):

        if isinstance(pipe_input, pd.DataFrame):
            return self.call_on_dataframe(pipe_input, **kwargs)

        elif isinstance(pipe_input, (pd.Series, dict)):
            return self.call_on_sample(pipe_input, **kwargs)

        elif isinstance(pipe_input, str):
            return self.call_on_single(pipe_input, **kwargs)

        return
