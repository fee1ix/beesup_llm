from beesup_llm import *
from beesup_llm.model_pipelines import *
from beesup_llm.dataset import *

from beesup_llm.injection import get_system_prompt, get_context

#from beesup_llm.injection.evaluator import Evaluator


# Selector Classes: Used to select which first n chunks will be included to the briefing
# input: ranking_df .. chunks sorted decreasingly by score + selection criteria
# output: list of booleans, True if chunk should be included, False otherwise

class RAGSelector(object):
    type='selector'

    @classmethod
    def from_ref(cls, type=None, **kwargs):
        kwargs.update(get_cls_attrs(cls))

        if PreTrimLimit.matches(type): return PreTrimLimit(**kwargs)
        if FitToknLimit.matches(type): return FitToknLimit(**kwargs)
        if FitCharLimit.matches(type): return FitCharLimit(**kwargs)
        if FitKneeScore.matches(type): return FitKneeScore(**kwargs)

        return cls(**kwargs)

    @classmethod
    def matches(cls, type=None, name=None):
        if type == cls.type: return True

        return False


class PreTrimLimit(RAGSelector):
    """
    Selects the first n chunks, reduces complexity for following operations
    """
    type='pre_trim_limit'

    def __init__(self, limit=100, **kwargs):
        self.limit=limit

    @staticmethod
    def add_feature(chunks_df: pd.DataFrame, **kwargs):
        return
    
    def get_mask(self, ranking_df: pd.DataFrame) -> list:
        return [True]*self.limit + [False]*(len(ranking_df)-self.limit)



class FitToknLimit(RAGSelector):
    """
    Selects chunks until the cumulative number of tokens reaches the limit
    """
    type='fit_tokn_limit'
    def __init__(self, limit=500):
        self.limit=limit
    
    @staticmethod
    def add_feature(chunks_df: pd.DataFrame, **kwargs):

        assert 'tokenizer' in kwargs, "tokenizer missing in kwargs"
        tokenizer=kwargs['tokenizer']

        if 'n_tokns' not in chunks_df.columns:
            chunks_df['n_tokns'] = chunks_df['chunk'].apply(lambda x: len(tokenizer(x)['input_ids']))
    
        return 
    
    def get_mask(self, ranking_df: pd.DataFrame) -> list:
        assert 'n_tokns' in ranking_df.columns, "n_tokns column missing in ranking_df"
        ranking_df['n_tokns_cumsum'] = ranking_df['n_tokns'].cumsum()
        return (ranking_df['n_tokns_cumsum']<self.limit).to_list()

    

class FitCharLimit(RAGSelector):
    """
    Selects chunks until the cumulative number of characters reaches the limit
    """
    type='fit_char_limit'
    def __init__(self, limit=500):
        self.limit=limit
    
    @staticmethod
    def add_feature(chunks_df: pd.DataFrame, **kwargs):
        if 'n_chars' not in chunks_df.columns:
            chunks_df['n_chars'] = chunks_df['chunk'].apply(len)
    
    def get_mask(self, ranking_df: pd.DataFrame) -> list:
        assert 'n_chars' in ranking_df.columns, "n_chars column missing in ranking_df"
        ranking_df['n_chars_cumsum'] = ranking_df['n_chars'].cumsum()
        return (ranking_df['n_chars_cumsum']<self.limit).to_list()



from kneed import KneeLocator
class FitKneeScore(RAGSelector):
    """
    Selects chunks with scores above the knee score
    """
    type='fit_knee_score'

    def __init__(self, curve="convex", direction="decreasing"):
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

class RAGPipeline(BaseDirectory):
    type='rag_pipeline'

    def __init__(
            self,
            ref=None,
            llm_ref=None,
            dataset_ref=None,
            selectors: list = [pre_trim_limit(limit=100),fit_tokn_limit(limit=1000), fit_knee_score()],
            **kwargs):
        super().__init__(ref, **kwargs)

        self._default_config=dict(
            emb_col='spo',
            chunk_col='spo',
            llm_config=dict(
                generation_config=dict(
                    max_new_tokens=4096,
                    max_time=1200,
                ),
            selector_configs=[]
            ),
        )

        self._config_key_order.extend(list(self._default_config.keys()))
        self._config_keys_to_exclude.extend(['llm_pipe','dataset','selectors'])

        self.update_config(self._default_config, overwrite_if_conflict=False)
        self.update_config_smart(kwargs)

        if self.is_spawned():
            if llm_ref==None: llm_ref=self.llm_config
            dataset_ref=self.dataset_config
            selectors=self.eval_configs


        self.selectors=selectors
        for selector in self.selectors:
            self.selector_configs.append()



            setattr(self, selector.__class__.__name__, selector.__dict__)

        if dataset_ref:
            dataset=BaseDataset.from_ref(dataset_ref)
            dataset_df=dataset.df

            if not dataset.parent_config: self.logger.warning('Dataset has no parent config')
            parent_df=BaseDataset(dataset.parent_config).df

            # add embedding combinations
            add_cols=[self.chunk_col,'source_name','attr_type','n_units','n_words']
            dataset_df=dataset_df.merge(parent_df[add_cols+[f"{self.emb_col}_emb"]], left_on='kidx', right_index=True, how='left')
            dataset_df.rename(columns={f"{self.emb_col}_emb":'emb', f"{self.chunk_col}":'chunk'}, inplace=True)
            self.df=dataset_df
   
        if llm_ref:
            self.llm_pipe=LanguageModelPipeline.from_ref(llm_ref)
            self.llm_pipe.update_config(self.llm_config)
            self.llm_pipe.update_config_smart(kwargs)
        
        self.add_selector_features()
    
    def add_selector_features(self):

        tokenizer=self.llm_pipe.get_inference_tokenizer()

        for selector in self.selectors:
            selector.add_feature(self.df, tokenizer=tokenizer)
 
        return

    def get_ranking_df(self, sample: pd.Series) -> pd.DataFrame:
        assert 'question_emb' in sample, "question_emb missing in sample"
        
        chunk_embs=np.vstack(self.df['emb'].values)
        question_emb=sample['question_emb']

        ranking_df=self.df.copy()
        ranking_df['score']=cosine_similarity(question_emb.reshape(1,-1),chunk_embs)[0]
        ranking_df=ranking_df.sort_values('score', ascending=False)

        return ranking_df

    def get_briefing_df(self, ranking_df: pd.DataFrame) -> pd.DataFrame:

        for selector in self.selectors:
            ranking_df[selector.__class__.__name__] = selector.get_mask(ranking_df)
        
        #select only rows where all selectors are True
        briefing_df=ranking_df[ranking_df[[s.__class__.__name__ for s in self.selectors]].all(axis=1)].copy()

        return briefing_df
    
    def add_briefing_df(self, pipe_df: pd.DataFrame):
        assert 'question_emb' in pipe_df, "question_emb missing in samples"

        chunk_embs=np.vstack(self.df['emb'].values)
        question_embs=np.vstack(pipe_df['question_emb'].values)

        score_matrix=cosine_similarity(question_embs,chunk_embs)
        index_matrix=np.argsort(-score_matrix, axis=1)
        score_matrix=-np.sort(-score_matrix, axis=1) #sort scores descending

        briefing_dfs=[]
        for i in range(len(pipe_df)):
            ranking_df=self.df.iloc[index_matrix[i]].copy()
            ranking_df['score']=score_matrix[i]
            briefing_dfs.append(self.get_briefing_df(ranking_df))
        pipe_df['briefing_df']=briefing_dfs

    def get_prompt_messages(self, sample, **kwargs):

        ranking_df=self.get_ranking_df(sample)
        briefing_df=self.get_briefing_df(ranking_df)

        prompt_messages=[]
        prompt_messages.append(dict(role='system', content=get_system_prompt(rag=True,**kwargs)))

        #user message with briefing chunks + question
        prompt_messages.append(dict(role='user', content=''))
        prompt_messages[-1]['content']+=get_context(briefing_df, chunk_col=self.chunk_col)

        prompt_messages[-1]['content']+="### QUESTION:\n"
        prompt_messages[-1]['content']+=sample['question']

        return prompt_messages
    
    def call_on_dataframe(self, df, **kwargs):
            
            pred_df=df.copy()
            pred_df['prompt_messages']=pred_df.apply(lambda x: self.get_prompt_messages(x), axis=1)
            pred_df=self.llm_pipe(pred_df, **kwargs)
            return pred_df
    
    def __call__(self, the_input, **kwargs):

        if isinstance(the_input, pd.DataFrame):
            self.call_on_dataframe(the_input, **kwargs)

        return self.get_prompt_messages(the_input)
