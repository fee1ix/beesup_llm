from beesup_llm import *
from ..toolkit.setup_utils import *
from ..toolkit.llm_utils import *

from beesup_llm.dataset import BaseDataset
#from beesup_llm.model import BaseModelWrap
import logging

import pandas as pd


class BaseEvaluation(BaseDirectory):
    type='evaluation'

    def __init__(self, ref=None, dataset_ref=None, **kwargs):

        super().__init__(ref,**kwargs)

        self._config_key_order.extend([])
        self._config_keys_to_exclude.extend(['df'])

        self._default_config=dict()
        self.update_attributes(self._default_config, overwrite=False)

        if isinstance(dataset_ref, pd.DataFrame):
            self.df=dataset_ref

        else:
            dataset=BaseDataset(dataset_ref)
            self.dataset_config=dataset.get_config()
            self.df=dataset.dataset_df

        self._default_config=dict()
        self._config_key_order.extend([k for k in self._default_config.keys() if k not in self._config_key_order])
        self.update_attributes(self._default_config, overwrite=False)
    







