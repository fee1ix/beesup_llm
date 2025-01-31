from beesup_llm import *
from beesup_llm.toolkit.setup_utils import *
from beesup_llm.injection.evaluation_utils import *

from beesup_llm.dataset import *
from beesup_llm.model_pipelines import *

import pickle
import pandas as pd


class EvaluationPipeline(BaseDirectory):
    type='evaluation_pipeline'

    def __init__(self, ref=None, llm_ref=None, samples_df=None,**kwargs):
        super().__init__(ref, **kwargs)

        self._default_config=dict(
            llm_config=dict(
                generation_config=dict(
                    stop_strings=[],
                )
            ) 
        )

        self._config_key_order.extend([k for k in self._default_config.keys() if k not in self._config_key_order])
        self._config_keys_to_exclude.extend([])

        self.update_config(self._default_config, overwrite_if_conflict=False)
        self.update_config_smart(kwargs)

        if llm_ref:
            self.llm_pipe=LanguageModelPipeline.from_ref(llm_ref)
            self.llm_pipe.update_config(self.llm_config)
            #self.llm_pipe.update_config(self._default_config['llm_config'])
            self.llm_pipe.update_config_smart(kwargs)
            self.llm_config=self.llm_pipe.get_config()
            #self.update_config(dict(llm_config=self.llm_pipe.get_config()), overwrite_if_conflict=False)
        

class MCQEvaluationPipeline(EvaluationPipeline):
    type='evaluation_pipeline'

    def __init__(self, ref=None, **kwargs):
        super().__init__(ref, **kwargs)

        self._default_config=dict()

        self._config_key_order.extend(list(self._default_config.keys()))
        self._config_keys_to_exclude.extend([])
        
        self.update_config(self._default_config, overwrite_if_conflict=True)
        self.update_config_smart(
            kwargs, 
            interpret_none_as_val=True, 
            overwrite_if_conflict=True, 
            allow_new_atomic_keys=False, 
            allow_new_nested_keys=False
        )

        


