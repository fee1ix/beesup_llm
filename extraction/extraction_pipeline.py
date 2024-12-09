from beesup_llm import *

from ..toolkit.setup_utils import *
from ..toolkit.llm_utils import *

from .extraction_utils import *


from beesup_llm.dataset import *
#from beesup_llm.model import *
from beesup_llm.model_pipelines import *


from datasets import Dataset


class ExtractionSample(object):
    
    def __init__(self, pred_completion=None, **kwargs):

        self.pred_completion=pred_completion

        for k,val in kwargs.items(): 
            setattr(self,k,val)
     
        assert hasattr(self, 'pred_completion'), "missing 'pred_completion'"
        self.parse_json(prefix='pred', exclude_none=True)
        self.parse_df(prefix='pred')

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
    
class ExtractionPipeline(BaseDirectory):
    type='extraction_pipeline'

    @classmethod
    def from_ref(cls, ref=None, **kwargs):
        kwargs.update(get_cls_attrs(cls))
        cls.logger.debug(f"{cls} ref={ref}, kwargs = {kwargs}\n")

        pre_config = get_config_from_ref(ref, **kwargs)
        
        return cls(ref=pre_config, **kwargs)

    def __init__(self, ref=None, llm_ref=None, **kwargs):
        super().__init__(ref, **kwargs)

        self._default_config=dict(
            use_extraction_prompt=True,
            use_few_shots=True,
            llm_config=dict(
                generation_config=dict(
                    stop_strings=['}\n```'],
                )
            )
        )

        self._config_key_order.extend([k for k in self._default_config.keys() if k not in self._config_key_order])
        self._config_keys_to_exclude.extend([])

        self.update_config(self._default_config, overwrite_if_conflict=False)
        self.update_config_smart(kwargs)

        if llm_ref:
            self.llm_pipe=LanguageModelPipeline.from_ref(llm_ref)
            self.llm_pipe.update_config(self._default_config['llm_config'])
            self.llm_pipe.update_config_smart(kwargs)
            self.llm_config=self.llm_pipe.get_config()
            #self.update_config(dict(llm_config=self.llm_pipe.get_config()), overwrite_if_conflict=False)

 
    def get_prompting_config(self, **kwargs):

        prompting_config=dict(
            use_extraction_prompt=self.use_extraction_prompt,
            use_few_shots=self.use_few_shots
        )
        kwargs=filter_kwargs(kwargs, prompting_config.keys())
        prompting_config.update(kwargs)

        return prompting_config

    @staticmethod
    def get_pred_parse_only(pred_completion, **kwargs):
        sample=ExtractionSample(pred_completion=pred_completion)
        sample.parse_df()
        return sample.pred_df

    def get_pred(self, report_passage, llm_pipe, **kwargs):

        llm_pipe.prepare_inference()
        prompt_messages=get_prompt_messages(report_passage, **self.get_prompting_config(**kwargs))

        pred_completion=''
        for new_token in llm_pipe.get_pipeline_stream(prompt_messages, **kwargs):
            pred_completion+=new_token
            print(new_token, end='', flush=True)
        
        return self.get_pred_parse_only(pred_completion,**kwargs)

    def prepare_df_for_completion(self, df, **kwargs):
        assert 'report_passage' in df.columns, "df must have 'prompt_messages' column"
        df['prompt_messages']=df['report_passage'].apply(lambda x: get_prompt_messages(x,**self.get_prompting_config(**kwargs)))
        return df
    
    def prepare_df_for_finetuning(self, df, **kwargs):
        assert 'gold_completion' in df.columns, "df must have 'gold_completion' column"
        df=self.prepare_df_for_completion(df, **kwargs)
        df['gold_message']=df['gold_completion'].apply(lambda x: [{'role':'assistant','content': x}])
        return df

    def get_ds_for_finetuning(self, df, tokenizer, **kwargs):
        assert 'prompt_messages' in df.columns, "df must have 'prompt_messages' column"
        assert 'gold_message' in df.columns, "df must have 'gold_message' column"
        ds=Dataset.from_list(df.apply(lambda x: prepare_sample_for_chat_finetuning(x, tokenizer),axis=1).to_list())
        return ds
    
    def get_ds_for_completion(self, df, tokenizer, **kwargs):
        assert 'prompt_messages' in df.columns, "df must have 'prompt_messages' column"
        ds=Dataset.from_list(df.apply(lambda x: prepare_sample_for_chat_completion(x, tokenizer),axis=1).to_list())
        return ds

    @staticmethod
    def get_pred_df_parse_only(df, **kwargs):
        assert 'pred_completion' in df.columns, "missing 'pred_completion' column"
        
        df[['pred_json','pred_is_valid','pred_is_empty']]=None,None,None
        for i,row in df.iterrows():
            try:
                sample=ExtractionSample(pred_completion=row['pred_completion'])
                df.at[i,'pred_json']=sample.pred_json
                df.at[i,'pred_is_valid']=sample.pred_is_valid
                df.at[i,'pred_is_empty']=sample.pred_is_empty

            except: pass
        
        return df

    def get_pred_df(self, df, llm_pipe, **kwargs):

        df = self.prepare_df_for_completion(df, **kwargs)
        llm_pipe.prepare_inference()

        df = llm_pipe.get_pred_df(df, **kwargs)
        self.llm_pipe._recent_generation_config=llm_pipe._recent_generation_config
        df = self.get_pred_df_parse_only(df)
        return df

    
    def __call__(self, the_input, llm_ref=None, **kwargs):

        if llm_ref:
            llm_pipe=LanguageModelPipeline.from_ref(llm_ref)
            llm_pipe.update_config(self._default_config['llm_config'])
    
        elif hasattr(self, 'llm_pipe'):
            llm_pipe=self.llm_pipe

        if isinstance(the_input, str): #input is a single report passage
            return self.get_pred(the_input, llm_pipe, **kwargs)
        
        elif isinstance(the_input, pd.DataFrame): #input is a dataframe containing a column 'report_passage'
            return self.get_pred_df(the_input, llm_pipe, **kwargs)

        elif isinstance(the_input, Dataset):
            self.logger.info("Dataset input detected")

        return



    

    






        

 
 


    
    



