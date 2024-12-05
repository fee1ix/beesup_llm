from beesup_llm import *
from beesup_llm.toolkit.llm_utils import *
from beesup_llm.toolkit.setup_utils import *

from beesup_llm.model.embedding_models import *
from beesup_llm.model.generation_models import *


import logging
import pandas as pd
import torch


class BaseModelWrap(BaseDirectory):

    type='model'

    @staticmethod
    def matches(ref):
        if getattr_or_key(ref, 'type') == ['model']: return True
        if GenModelWrap.matches(ref): return True
        if EmbModelWrap.matches(ref): return True
        return False

    @staticmethod
    def get_subconfig(ref):

        gen_model=getattr_or_key(ref,'gen_model_config',False)
        emb_model=getattr_or_key(ref,'emb_model_config',False)

        if gen_model and emb_model: raise ValueError("Both gen_model and emb_model are present in the config.")
        if gen_model: return gen_model
        if emb_model: return emb_model
        else: return None
            
    @classmethod
    def from_ref(cls, ref=None, **kwargs):
        kwargs.update(get_cls_attrs(cls))
        cls.logger.debug(f"{cls} ref={ref}, kwargs = {kwargs}\n")

        if isinstance(ref, cls): return ref

        pre_config = get_config_from_ref(ref, **kwargs)

        
        #if hasattr(ref,'model'): kwargs['model']=ref.model
            
        if GenModelWrap.matches(pre_config):
            return GenModelWrap(ref=pre_config, **kwargs)
        
        if EmbModelWrap.matches(pre_config):
            return EmbModelWrap.from_ref(ref=pre_config, **kwargs)

        return cls(ref=pre_config, **kwargs)
    
    def __init__(self, ref=None, **kwargs):
        self.logger.debug(f"{self.__class__} ref={ref}, kwargs = {kwargs}\n")
        super().__init__(ref, **kwargs)
        self._config_key_order.extend(['name_or_path'])
        self._config_keys_to_exclude.extend(['model'])#

        self._default_config=dict()
        self.update_config(self._default_config, overwrite_if_conflict=False)
    
    def get_model(self):

        model=getattr(self, 'model', None)
        if model is None:
            self.load_model()
            model=self.model
            del self.model
            return model
        
        else:
            return self.model






