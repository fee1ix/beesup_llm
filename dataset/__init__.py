
from beesup_llm import *
from ..toolkit.setup_utils import *

import pandas as pd
from datasets import Dataset
import logging



class BaseDataset(BaseDirectory):

    def __init__(self, config=None, parent_lab=None, dataset_df=None):

        if config is None: #load default config from repository
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config = os.path.join(current_dir, 'base_dataset_config.yaml')
        
        super().__init__(config, parent_lab)

        if os.path.exists(f'{self.path}/dataset_df.pkl'):
            self.dataset_df=pd.read_pickle(f'{self.path}/dataset_df.pkl')

            if dataset_df is not None: raise ValueError("dataset_df already exists")


        elif dataset_df is not None:
            self.dataset_df=dataset_df
        

    
    def spawn(self):
        assert hasattr(self, 'dataset_df'), "Dataset must be assigned before spawning"
        assert isinstance(self.dataset_df, pd.DataFrame), "Dataset must be a pandas DataFrame."

        required_cols=['prompt','gold_completion','prompt_messages','gold_message']

        for required_col in required_cols:
            if required_col not in self.dataset_df.columns:
                self.dataset_df[required_col]=None

        self.dataset_df.to_pickle(f"{self.path}/dataset_df.pkl")
        set_config(self.get_config)

        logging.info(f"{self.name.upper()} spawned at {self.path}")

    
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

    def arrange(self, tokenizer):

        self.logger.info(f"{self.name.upper()} START")

        for required_col in ['prompt','gold_completion','prompt_messages','gold_message']:
            if required_col not in self.dataset_df.columns:
                self.dataset_df[required_col]=None

        train_ds,eval_ds,test_ds=None,None,None

        if 'train' in self.dataset_df['split'].values:
            train_ds=Dataset.from_list(self.dataset_df[self.dataset_df.split=='train'].apply(lambda x: self.arrange_sample(x, tokenizer),axis=1).to_list())

        if 'eval' in self.dataset_df['split'].values:
            eval_ds=Dataset.from_list(self.dataset_df[self.dataset_df.split=='eval'].apply(lambda x: self.arrange_sample(x, tokenizer),axis=1).to_list())
            #test_ds=Dataset.from_list(dataset_df[dataset_df.split=='eval'].apply(arrange_sample,axis=1).to_list()[:2])

        if 'test' in self.dataset_df['split'].values:
            test_ds=Dataset.from_list(self.dataset_df[self.dataset_df.split=='test'].apply(lambda x: self.arrange_sample(x, tokenizer),axis=1).to_list())
        
        if train_ds: self.logger.info(f'train_ds: {len(train_ds)} samples')
        if eval_ds: self.logger.info(f'eval_ds: {len(eval_ds)} samples')
        if test_ds: self.logger.info(f'test_ds: {len(test_ds)} samples')

        return train_ds,eval_ds,test_ds

        






 












