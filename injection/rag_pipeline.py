from beesup_llm import *
from beesup_llm.model_pipelines import *
from beesup_llm.dataset import *

from beesup_llm.injection import get_system_prompt

# Selector Classes: Used to select chunks presented to the language model
class fit_tokn_limit:
    def __init__(self, limit=500):
        self.limit=limit
    
    def __call__(self, ranking_df):
        ranking_df=ranking_df.copy()
        assert 'n_tokns' in ranking_df.columns, "n_tokns column missing in ranking_df"
        ranking_df['n_tokns_cumsum'] = ranking_df['n_tokns'].cumsum()
        return (ranking_df['n_tokns_cumsum']<self.limit).to_list()

from kneed import KneeLocator
class fit_knee_score:
    def __call__(self, ranking_df):
        ranking_df=ranking_df.copy()
        assert 'score' in ranking_df.columns, "score column missing in ranking_df"
        knee_index = KneeLocator(list(range(len(ranking_df))), ranking_df['score'], curve="convex", direction="decreasing").knee
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
            selectors: list = [fit_tokn_limit(limit=1000), fit_knee_score()],
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
            ),
        )

        self._config_key_order.extend(list(self._default_config.keys()))
        self._config_keys_to_exclude.extend(['llm_pipe','dataset','selectors'])

        self.update_config(self._default_config, overwrite_if_conflict=False)
        self.update_config_smart(kwargs)

        self.selectors=selectors

        if dataset_ref:
            dataset=BaseDataset.from_ref(dataset_ref)
            dataset_df=dataset.df

            if not dataset.parent_config: self.logger.warning('Dataset has no parent config')
            parent_df=BaseDataset(dataset.parent_config).df

            # add embedding combinations
            add_cols=[self.chunk_col,'source_name','attr_type','n_units','n_words']
            dataset_df=dataset_df.merge(parent_df[add_cols+[f"{self.emb_col}_emb"]], left_on='kidx', right_index=True, how='left')
            dataset_df.rename(columns={f"{self.emb_col}_emb":'emb'}, inplace=True)
            self.df=dataset_df

            print(self.df.columns)
            
        
        if llm_ref:
            self.llm_pipe=LanguageModelPipeline.from_ref(llm_ref)
            self.llm_pipe.update_config(self.llm_config)
            self.llm_pipe.update_config_smart(kwargs)
        
        self.add_df_features()
    

    def add_df_features(self):
        
        df=self.df.copy()
        tokenizer=self.llm_pipe.get_inference_tokenizer()
        
        if 'n_tokns' not in df.columns:
            df['n_tokns'] = df[self.chunk_col].apply(lambda x: len(tokenizer(x)['input_ids']))

        if 'n_chars' not in df.columns:
            df['n_chars'] = df[self.chunk_col].apply(len)
        
        self.df = df
        return


    def get_ranking_df(self, sample):

        chunk_embs=np.vstack(self.df['emb'].values)
        question_emb=sample['question_emb']

        ranking_df=self.df.copy()
        ranking_df['score']=cosine_similarity(question_emb.reshape(1,-1),chunk_embs)[0]
        ranking_df=ranking_df.sort_values('score', ascending=False)

        return ranking_df

    def get_prompt_messages(self, sample):

        ranking_df=self.get_ranking_df(sample)

        for selector in self.selectors:
            ranking_df[selector.__class__.__name__] = selector(ranking_df)
        
        briefing_df=ranking_df[ranking_df[[s.__class__.__name__ for s in self.selectors]].all(axis=1)].copy()

        prompt_messages=[]

        #system message
        prompt_messages.append(dict(role='system', content=get_system_prompt()))
        prompt_messages[-1]['content']+="""
    You have access to relevant knowledge chunks. \
    Your goal is to answer the user's question strictly based on the provided context. \
    If the answer is not present in the context, state "I don't know". \
    Do not attempt to answer using outside knowledge.
    """.strip()
        
        #user message with briefing chunks + question
        
        prompt_messages.append(dict(role='user', content=''))
        prompt_messages[-1]['content']+="### Context:\n"
        for _,row in briefing_df.iterrows():
            prompt_messages[-1]['content']+=f"{row[self.chunk_col]}\n\n"

        prompt_messages[-1]['content']+="### User Question:\n"
        prompt_messages[-1]['content']+=sample['question']

        return prompt_messages
    

    def __call__(self, the_input, **kwargs):

        if isinstance(the_input, pd.DataFrame):

            pred_df=the_input.copy()
            pred_df['prompt_messages']=pred_df.apply(lambda x: self.get_prompt_messages(x), axis=1)
            pred_df=self.llm_pipe(pred_df, **kwargs)
            return pred_df

        return self.get_prompt_messages(the_input)




    #def __call__(self, question, llm_ref=None, **kwargs):

