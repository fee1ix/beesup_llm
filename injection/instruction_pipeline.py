from beesup_llm import *
from beesup_llm.toolkit.setup_utils import *

from beesup_llm.injection.taxomizer import *

from beesup_llm.injection.instruction_utils import *

from beesup_llm.dataset import *
from beesup_llm.model_pipelines import *

import pickle
import pandas as pd


class InstructionPipeline(BaseDirectory):
    type='instruction_pipeline'

    def __init__(self, ref=None, tax_ref=None,**kwargs):
        super().__init__(ref, **kwargs)

        self._default_config=dict(
            tax_config=dict(),
        )


        self._config_key_order.extend(list(self._default_config.keys()))
        self._config_keys_to_exclude.extend(['tax_pipe'])

        self.update_config(self._default_config, overwrite_if_conflict=False)
        self.update_config_smart(kwargs)

        if self.is_spawned():
            tax_ref=self.tax_config

        if tax_ref:
            self.tax_pipe=TaxomizingPipeline(tax_ref)
            self.tax_pipe.update_config(self._default_config['tax_config'])
            self.tax_pipe.update_config_smart(kwargs)
            self.tax_config=self.tax_pipe.get_config()
    
    

        

class SimpleInstructionPipeline(InstructionPipeline):

    def __init__(self, ref=None, **kwargs):
        super().__init__(ref, **kwargs)
    





