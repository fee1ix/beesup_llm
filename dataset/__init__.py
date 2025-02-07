
from beesup_llm import *
from ..toolkit.setup_utils import *

from datasets import Dataset
import pandas as pd
import logging


class BaseDataset(BaseDirectory):
    type='dataset'

    @classmethod
    def from_ref(cls, ref=None, **kwargs):
        kwargs.update(get_cls_attrs(cls))
        cls.logger.debug(f"{cls} ref={ref}, kwargs = {kwargs}\n")

        pre_config = get_config_from_ref(ref, **kwargs)

        #if hasattr(ref,'df'): kwargs['df']=ref.df

        return cls(ref=pre_config, **kwargs)

    @classmethod
    def from_df(cls, df, **kwargs):
        kwargs.update(get_cls_attrs(cls))
        cls.logger.debug(f"{cls} df={df}, kwargs = {kwargs}\n")

        return cls(df=df, **kwargs)

    def __init__(self, ref=None, df=None, emb_model_ref=None, parent_ref=None, **kwargs):
        super().__init__(ref, **kwargs)

        self._default_config=dict(
            remarks=None,
            n_rows=None,
            emb_model_config=None,
            parent_config=None,
        )
        self._config_key_order.extend([k for k in self._default_config.keys() if k not in self._config_key_order])
        self._config_keys_to_exclude.extend(['df','dataset_df','emb_model'])

        self.update_config(self._default_config, overwrite_if_conflict=False)
        self.update_config_smart(
            kwargs, 
            interpret_none_as_val=True, 
            overwrite_if_conflict=True, 
            allow_new_atomic_keys=False, 
            allow_new_nested_keys=False
        )

        if os.path.exists(f'{self._path}/df.pkl'):
            self.df=pd.read_pickle(f'{self._path}/df.pkl')
            if df is not None: raise ValueError("df already exists")
        
        if os.path.exists(f'{self._path}/dataset_df.pkl'):
            self.df=pd.read_pickle(f'{self._path}/dataset_df.pkl')
            if df is not None: raise ValueError("dataset_df already exists")

        if df is not None: self.set_df(df)
  
        if emb_model_ref is not None:
            from beesup_llm.model import EmbModelWrap
            self.emb_model_config=EmbModelWrap.from_ref(emb_model_ref).get_config()
        
        if parent_ref is not None:
            parent_config=BaseDataset(parent_ref).get_config()
            if 'parent_config' in parent_config: del parent_config['parent_config']
            self.parent_config=parent_config
    
    def set_df(self, df=None):
        if isinstance(df, pd.DataFrame):
            self.df=df.copy()
        
        self.df.reset_index(drop=True, inplace=True)
        self.n_rows=len(self.df)
        return


    def get_df_splits(self, splits='train'):
        if isinstance(splits, str): splits=[splits]
        return self.df[self.df.split.isin(splits)].copy()

    def spawn(self):
        super().spawn()
        assert hasattr(self, 'df'), "Dataset must be assigned before spawning"
        assert isinstance(self.df, pd.DataFrame), "Dataset must be a pandas DataFrame."
        self.df.to_pickle(f"{self._path}/df.pkl")

        
  
    # def arrange_sample(self, sample, tokenizer):

    #     if isinstance(sample.get('prompt_messages'),list) and isinstance(sample.get('gold_message'),list):
    #         all_messages=sample['prompt_messages']+sample['gold_message']

    #         prompt_ids=tokenizer.apply_chat_template(sample['prompt_messages'],tokenize=True)
    #         prompt_len=len(prompt_ids)

    #         input_ids=tokenizer.apply_chat_template(sample['prompt_messages']+sample['gold_message'],tokenize=True)
            
    #         input_text=tokenizer.apply_chat_template(all_messages,tokenize=False)
    #         inputs=tokenizer.apply_chat_template(all_messages,return_dict=True)

    #         inputs['labels']=prompt_len*[-100]+input_ids[prompt_len:]
            
        
    #     elif pd.notna(sample.get('prompt')) and pd.notna(sample.get('gold_completion')):

    #         # if (not sample['prompt'].endswith('\n')) and (not sample['gold_completion'].startswith('\n')):
    #         #     warnings.warn('No newline between prompt and gold_completion --> Newline added!')
    #         #     input_text=sample['prompt']+'\n'+sample['gold_completion']
            
    #         input_text=sample['prompt']+sample['gold_completion']

    #         prompt_ids=tokenizer.encode(sample['prompt'],add_special_tokens=True)
    #         prompt_len=len(prompt_ids)

    #         input_ids=tokenizer.encode(sample['prompt']+sample['gold_completion'],add_special_tokens=True)#+[tokenizer.eos_token_id]

    #         inputs=tokenizer(input_text)
    #         inputs['labels']=prompt_len*[-100]+input_ids[prompt_len:]

    #         input_text=tokenizer.decode(inputs['input_ids'])
        

    #     elif isinstance(sample.get('prompt_messages'),list) and sample.get('split') in ['eval','test']:
    #         input_text=tokenizer.apply_chat_template(sample['prompt_messages'],tokenize=False)
    #         inputs=tokenizer.apply_chat_template(sample['prompt_messages'],return_dict=True)

    #     elif pd.notna(sample.get('prompt')) and sample.get('split') in ['eval','test']:
            
    #         inputs=tokenizer(sample['prompt'])
    #         input_text=tokenizer.decode(inputs['input_ids'])
        
    #     else:
    #         raise Warning("undefined sample format")

    #     return {**inputs}
    #     #return {'text':input_text,**inputs}

    # def arrange(self, tokenizer, df=None):

    #     if not hasattr(tokenizer, 'apply_chat_template'):
    #         raise AttributeError("The tokenizer does not have the method 'apply_chat_template'")

    #     if df is None:
    #         df=self.df.copy()


    #     self.logger.info(f"{self.name.upper()} START")

    #     for required_col in ['prompt','gold_completion','prompt_messages','gold_message']:
    #         if required_col not in df.columns:
    #             df[required_col]=None

    #     train_ds,eval_ds,test_ds=None,None,None

    #     if 'train' in df['split'].values:
    #         train_ds=Dataset.from_list(df[df.split=='train'].apply(lambda x: self.arrange_sample(x, tokenizer),axis=1).to_list())

    #     if 'eval' in df['split'].values:
    #         eval_ds=Dataset.from_list(df[df.split=='eval'].apply(lambda x: self.arrange_sample(x, tokenizer),axis=1).to_list())
    #         #test_ds=Dataset.from_list(df[df.split=='eval'].apply(arrange_sample,axis=1).to_list()[:2])

    #     if 'test' in df['split'].values:
    #         test_ds=Dataset.from_list(df[df.split=='test'].apply(lambda x: self.arrange_sample(x, tokenizer),axis=1).to_list())
        
    #     if train_ds: self.logger.info(f'train_ds: {len(train_ds)} samples')
    #     if eval_ds: self.logger.info(f'eval_ds: {len(eval_ds)} samples')
    #     if test_ds: self.logger.info(f'test_ds: {len(test_ds)} samples')

    #     return train_ds,eval_ds,test_ds











