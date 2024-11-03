from beesup_llm import *
from ..toolkit.setup_utils import *
from ..toolkit.llm_utils import *

#from beesup_llm.dataset import BaseDataset
#from beesup_llm.model import BaseModelWrap
import logging


class BaseEvaluation(BaseDirectory):

    def __init__(self, ref=None, refs=[]):

        self.type='evaluation'
        super().__init__(ref)

        self._config_key_order.extend([])
        self._config_keys_to_exclude.extend([])

        if hasattr(self, 'refs'):
            refs = self.refs
        
        for i,ref in enumerate(refs):
            if hasattr_or_key(ref, 'path'):
                ref = getattr_or_key(ref, 'path')
   
            refs[i]=BaseDirectory(ref)
        
        self.refs=refs

        self._default_config=dict()
        self._config_key_order.extend([k for k in self._default_config.keys() if k not in self._config_key_order])
        self.update_attributes(self._default_config, overwrite=False)
    







