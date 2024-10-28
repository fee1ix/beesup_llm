from beesup_llm import *
from ..toolkit.setup_utils import *


import pytz
import datetime
TIMEZONE = pytz.timezone('Europe/Berlin')

from beesup_llm.dataset import BaseDataset
from beesup_llm.model import BaseModelWrap
import pandas as pd
import logging

class BaseTest(BaseDirectory):

    def __init__(self, ref=None, dataset_ref=None, model_ref=None):

        self.type='test'
        super().__init__(ref)

        self._config_key_order.extend([])
        self._config_keys_to_exclude.extend(['dataset_df'])

        self._default_config=dict(
            done = False,
            batch_size = 4,
            use_dataset_splits=['test'],
            generation_config=dict(
                do_sample=False,
            ),
        )

        self.update_attributes(self._default_config, overwrite=False)

        if hasattr(self, 'dataset_config'):
            dataset_ref = self.dataset_config
        
        if hasattr(self, 'model_config'):
            model_ref = self.model_config

        self._dataset=BaseDataset(dataset_ref)
        self.dataset_df=self.prepare_dataset_df(self._dataset.dataset_df)
        self.dataset_config=self._dataset.get_config()

        self._modelwrap=BaseModelWrap(model_ref)
        self.model_config=self._modelwrap.get_config()
    
    def spawn(self):

        if not os.path.exists(f'{self.path}'):
            os.makedirs(f'{self.path}', exist_ok=False)

        set_config(self.get_config())
        logging.info(f"{self.name.upper()} spawned at {self.path}")

    def get_inference_df(self,inference_outputs):

        inference_data=[]
        for i in range(len(inference_outputs['all_input_ids'])):
            row=dict()

            for col in ['all_input_ids', 'all_label_ids', 'all_all_ids', 'all_losses']:
                if isinstance(inference_outputs[col],type(None)): continue

                row[col[4:]]=inference_outputs[col][i]
            
            inference_data.append(row)

        inference_df=pd.DataFrame(inference_data)
        return inference_df

    def prepare_dataset_df(self,dataset_df):

        dataset_df.loc[~dataset_df.split.isin(self.use_dataset_splits),'split']='ignore'
        dataset_df.loc[dataset_df.split.isin(self.use_dataset_splits),'split']='test'

        return dataset_df

    def run(self, **kwargs):

        self.logger.info(f"Running {self.name.upper()}")
        self.datetime_start=datetime.datetime.now(TIMEZONE)

        if self.done:
            self.logger.warning(f"{self.name} already completed")
            return

        if not hasattr(self._modelwrap,'model'):
            self._modelwrap.load_model()

        if not hasattr(self._modelwrap,'inference_tokenizer'):
            self._modelwrap.load_inference_tokenizer()


        from torch.utils.data import DataLoader
        from transformers import DataCollatorForSeq2Seq

        data_collator=DataCollatorForSeq2Seq(
            model=self._modelwrap.model,
            tokenizer=self._modelwrap.inference_tokenizer,
            #padding='longest',
            label_pad_token_id =-100
            )
        
        _, _, test_ds = self._dataset.arrange(
            self._modelwrap.inference_tokenizer,
            dataset_df=self.dataset_df
            )

        dataloader=DataLoader(
            test_ds,
            batch_size=kwargs.get('batch_size',self.batch_size),
            collate_fn=data_collator,
            )
        
        inference_outputs=self._modelwrap.inference_loop(dataloader, **self.generation_config)
        inference_df=self.get_inference_df(inference_outputs)

        inference_df.to_pickle(f"{self.path}/inference_df.pkl")

        self.logger.info(f"{self.name.upper()} completed")
        self.datetime_end=datetime.datetime.now(TIMEZONE)
        self.done=True
        set_config(self.get_config())
        return

            
            













