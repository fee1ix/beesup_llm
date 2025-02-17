from beesup_llm import *
from beesup_llm.model_pipelines import *
from beesup_llm.dataset import *


class RAGPipeline(BaseDirectory):
    type='rag_pipeline'

    def __init__(self, ref=None, llm_ref=None, emb_ref=None, dataset_ref=None, **kwargs):
        super().__init__(ref, **kwargs)

        self._default_config=dict(
            emb_col='spo',
            max_briefing_tokens=1000,
            llm_config=dict(
                generation_config=dict(
                    max_new_tokens=4096,
                    max_time=1200,
                ),
            ),
        )

        self._config_key_order.extend(list(self._default_config.keys()))
        self._config_keys_to_exclude.extend(['llm_pipe','dataset'])

        self.update_config(self._default_config, overwrite_if_conflict=False)
        self.update_config_smart(kwargs)


        if dataset_ref:
            dataset=BaseDataset.from_ref(dataset_ref)
            dataset_df=dataset.df

            if not dataset.parent_config: self.logger.warning('Dataset has no parent config')

            parent_df=BaseDataset(dataset.parent_config).df

            # add embedding combinations
            add_cols=['s','p','o','source_name','attr_type','n_units','n_words']
            dataset_df=dataset_df.merge(parent_df[add_cols+[f"{self.emb_col}_emb"]], left_on='kidx', right_index=True, how='left')
            dataset_df.rename(columns={f"{self.emb_col}_emb":'emb'}, inplace=True)
            self.df=dataset_df
        
        if llm_ref:
            self.llm_pipe=LanguageModelPipeline.from_ref(llm_ref)
            self.llm_pipe.update_config(self.llm_config)
            self.llm_pipe.update_config_smart(kwargs)
        


    #def __call__(self, question, llm_ref=None, **kwargs):

