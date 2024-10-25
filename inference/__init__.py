from beesup_llm import *
from ..toolkit.setup_utils import *

from ..dataset import BaseDataset

import logging

class BaseTest(BaseDirectory):

    def __init__(self, ref=None, dataset_ref=None, model_ref=None):

        self.type='test'
        super().__init__(ref)

        self._config_key_order.extend([])
        self._config_keys_to_exclude.extend([])

        self.dataset=BaseDataset(dataset_ref)
        #self.model=BaseModel()




