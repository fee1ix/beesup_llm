from beesup_llm import *

from ..toolkit.setup_utils import *
from ..toolkit.llm_utils import *


from beesup_llm.model import *
from beesup_llm.dataset import *

from .extraction_utils import *

class ExtractionPipeline(BaseDirectory):
    type='extraction_pipeline'

    def __init__(self, ref=None, model_ref=None, **kwargs):

        super().__init__(ref, **kwargs)

        self._config_key_order.extend([])
        self._config_keys_to_exclude.extend(['modelwrap'])

        self._default_config=dict(
            use_extraction_prompt=True,
            use_few_shots=True
        )
        self.update_attributes(self._default_config, overwrite=False)

        # if dataset_ref is not None:
        #     if isinstance(dataset_ref, pd.DataFrame):
        #         self.df=dataset_ref

        #     else:
        #         dataset=BaseDataset(dataset_ref)
        #         self.dataset_config=dataset.get_config()
        #         self.df=dataset.dataset_df

        if model_ref is not None:
            self.modelwrap=GenModelWrap.from_ref(model_ref)
            self.gen_model_config=self.modelwrap.get_config()
            #self.tokenizer=self.modelwrap.get_inference_tokenizer()
    


    def process_completion(self, pred_completion):

        raw_df,tab_df=parse_completion(pred_completion)

        self.logger.info(f"is_valid: {parse_completion.is_valid}")
        self.logger.info(f"is_empty: {parse_completion.is_empty}")

    #def apply_report_passage(self, report_passage, **kwargs):

    def prepare_dataset(self, df, **kwargs):

        prompting_config=dict(
            use_extraction_prompt=self.use_extraction_prompt,
            use_few_shots=self.use_few_shots
        )
        prompting_config.update(kwargs)

        from datasets import Dataset

        assert 'report_passage' in df.columns, "df must have 'report_passage' column"
        df['prompt_messages']=df['report_passage'].apply(lambda x: get_prompt_messages(x,**prompting_config))
        ds=Dataset.from_list(df.apply(lambda x: prepare_sample(x, self.modelwrap.get_inference_tokenizer()),axis=1).to_list())

        return ds


    def apply_df(self, df, **kwargs):

        ds = self.prepare_dataset(df, **kwargs)
        self._ds=ds

        generation_outputs=self.modelwrap.generation_loop(ds,**kwargs)

        self._generation_outputs=generation_outputs
        generation_df=to_outputs_df(generation_outputs,tokenizer=self.modelwrap.get_inference_tokenizer())

        self._generation_df=generation_df

        df['pred_completion']=generation_df['pred_completion'].values

        return df

    
    def __call__(self, passages_df, **kwargs):

        #possible inputs:
        # - report passage
        # - dataframe with report passages

        _kwargs=dict(
            use_extraction_prompt=self.use_extraction_prompt,
            use_few_shots=self.use_few_shots
        )
        _kwargs.update(kwargs)

        prompt_messages=get_prompt_messages(report_passage, **kwargs)
        outputs=self.modelwrap(prompt_messages,**kwargs)




        # #prompt_ids=self.get_prompt_ids(report_passage, **kwargs)
        # outputs=self.modelwrap.inference_step({'input':prompt_ids})

        return outputs
    

    #def __call__by_report_passage(self, report_passage,**kwargs):


    
    #def __call__by_df(self, df, **kwargs):


    

    






        

 
 


    
    



