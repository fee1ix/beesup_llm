
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

        #if hasattr(ref,'dataset_df'): kwargs['dataset_df']=ref.dataset_df

        return cls(ref=pre_config, **kwargs)

    @classmethod
    def from_df(cls, dataset_df, **kwargs):
        kwargs.update(get_cls_attrs(cls))
        cls.logger.debug(f"{cls} dataset_df={dataset_df}, kwargs = {kwargs}\n")

        return cls(dataset_df=dataset_df, **kwargs)

    def __init__(self, ref=None, dataset_df=None, emb_model_ref=None, parent_ref=None, **kwargs):
        super().__init__(ref, **kwargs)
        self._config_key_order.extend([])
        self._config_keys_to_exclude.extend(['dataset_df','emb_model'])

        if os.path.exists(f'{self.path}/dataset_df.pkl'):
            self.dataset_df=pd.read_pickle(f'{self.path}/dataset_df.pkl')
            if dataset_df is not None: raise ValueError("dataset_df already exists")

        elif dataset_df is not None:
            self.dataset_df=dataset_df

        if emb_model_ref is not None:
            from beesup_llm.model import EmbModelWrap
            self.emb_model_config=EmbModelWrap.from_ref(emb_model_ref).get_config()
        
        if parent_ref is not None:
            parent_config=BaseDataset(parent_ref).get_config()
            if 'parent_config' in parent_config: del parent_config['parent_config']
            self.parent_config=parent_config
    
    def get_df_splits(self, splits='train'):
        if isinstance(splits, str): splits=[splits]
        return self.dataset_df[self.dataset_df.split.isin(splits)].copy()


    def spawn(self):
        assert hasattr(self, 'dataset_df'), "Dataset must be assigned before spawning"
        assert isinstance(self.dataset_df, pd.DataFrame), "Dataset must be a pandas DataFrame."

        if not os.path.exists(f'{self.path}'):
            os.makedirs(f'{self.path}', exist_ok=False)

        required_cols=[]
        #required_cols=['prompt','gold_completion','prompt_messages','gold_message']

        for required_col in required_cols:
            if required_col not in self.dataset_df.columns:
                self.dataset_df[required_col]=None
        
        self.dataset_df.reset_index(drop=True, inplace=True)
        self.dataset_df.to_pickle(f"{self.path}/dataset_df.pkl")
        set_config(self.get_config())

        logging.info(f"{self.name.upper()} spawned at {self.path}")
  
    #def get_df(self):








    def arrange_sample(self, sample, tokenizer):

        if isinstance(sample.get('prompt_messages'),list) and isinstance(sample.get('gold_message'),list):
            all_messages=sample['prompt_messages']+sample['gold_message']

            prompt_ids=tokenizer.apply_chat_template(sample['prompt_messages'],tokenize=True)
            prompt_len=len(prompt_ids)

            input_ids=tokenizer.apply_chat_template(sample['prompt_messages']+sample['gold_message'],tokenize=True)
            
            input_text=tokenizer.apply_chat_template(all_messages,tokenize=False)
            inputs=tokenizer.apply_chat_template(all_messages,return_dict=True)

            inputs['labels']=prompt_len*[-100]+input_ids[prompt_len:]
            
        
        elif pd.notna(sample.get('prompt')) and pd.notna(sample.get('gold_completion')):

            # if (not sample['prompt'].endswith('\n')) and (not sample['gold_completion'].startswith('\n')):
            #     warnings.warn('No newline between prompt and gold_completion --> Newline added!')
            #     input_text=sample['prompt']+'\n'+sample['gold_completion']
            
            input_text=sample['prompt']+sample['gold_completion']

            prompt_ids=tokenizer.encode(sample['prompt'],add_special_tokens=True)
            prompt_len=len(prompt_ids)

            input_ids=tokenizer.encode(sample['prompt']+sample['gold_completion'],add_special_tokens=True)#+[tokenizer.eos_token_id]

            inputs=tokenizer(input_text)
            inputs['labels']=prompt_len*[-100]+input_ids[prompt_len:]

            input_text=tokenizer.decode(inputs['input_ids'])
        

        elif isinstance(sample.get('prompt_messages'),list) and sample.get('split') in ['eval','test']:
            input_text=tokenizer.apply_chat_template(sample['prompt_messages'],tokenize=False)
            inputs=tokenizer.apply_chat_template(sample['prompt_messages'],return_dict=True)

        elif pd.notna(sample.get('prompt')) and sample.get('split') in ['eval','test']:
            
            inputs=tokenizer(sample['prompt'])
            input_text=tokenizer.decode(inputs['input_ids'])
        
        else:
            raise Warning("undefined sample format")

        return {**inputs}
        #return {'text':input_text,**inputs}

    def arrange(self, tokenizer, dataset_df=None):

        if not hasattr(tokenizer, 'apply_chat_template'):
            raise AttributeError("The tokenizer does not have the method 'apply_chat_template'")

        if dataset_df is None:
            dataset_df=self.dataset_df.copy()


        self.logger.info(f"{self.name.upper()} START")

        for required_col in ['prompt','gold_completion','prompt_messages','gold_message']:
            if required_col not in dataset_df.columns:
                dataset_df[required_col]=None

        train_ds,eval_ds,test_ds=None,None,None

        if 'train' in dataset_df['split'].values:
            train_ds=Dataset.from_list(dataset_df[dataset_df.split=='train'].apply(lambda x: self.arrange_sample(x, tokenizer),axis=1).to_list())

        if 'eval' in dataset_df['split'].values:
            eval_ds=Dataset.from_list(dataset_df[dataset_df.split=='eval'].apply(lambda x: self.arrange_sample(x, tokenizer),axis=1).to_list())
            #test_ds=Dataset.from_list(dataset_df[dataset_df.split=='eval'].apply(arrange_sample,axis=1).to_list()[:2])

        if 'test' in dataset_df['split'].values:
            test_ds=Dataset.from_list(dataset_df[dataset_df.split=='test'].apply(lambda x: self.arrange_sample(x, tokenizer),axis=1).to_list())
        
        if train_ds: self.logger.info(f'train_ds: {len(train_ds)} samples')
        if eval_ds: self.logger.info(f'eval_ds: {len(eval_ds)} samples')
        if test_ds: self.logger.info(f'test_ds: {len(test_ds)} samples')

        return train_ds,eval_ds,test_ds











