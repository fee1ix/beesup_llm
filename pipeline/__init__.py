from beesup_llm import *
from ..toolkit.setup_utils import *
from beesup_llm.dataset import BaseDataset

import pandas as pd

class BasePipeline(BaseDirectory):
    type='pipeline'

    def __init__(self, ref=None, dataset_ref=None, model_ref=None, **kwargs):

        super().__init__(ref, **kwargs)
        self._config_key_order.extend([])
        self._config_keys_to_exclude.extend(['df'])

        self._default_config=dict()
        self.update_attributes(self._default_config, overwrite=False)
        
        if dataset_ref is not None:
            if isinstance(dataset_ref, pd.DataFrame):
                self.df=dataset_ref

            else:
                dataset=BaseDataset(dataset_ref)
                self.dataset_config=dataset.get_config()
                self.df=dataset.dataset_df

        if model_ref is not None:
            if isinstance(dataset_ref, pd.DataFrame):
                self.df=dataset_ref

            else:
                dataset=BaseDataset(dataset_ref)
                self.dataset_config=dataset.get_config()
                self.df=dataset.dataset_df



    
