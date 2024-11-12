from .extraction_pipeline import *

from .evaluation_utils import *

class EvaluationPipeline(ExtractionPipeline):
    type='evaluation_pipeline'

    def __init__(self, ref=None, **kwargs):

        super().__init__(ref,  **kwargs)

        self._config_key_order.extend([])
        self._config_keys_to_exclude.extend([])

        self._default_config=dict()
        self.update_attributes(self._default_config, overwrite=False)


    # def __call__(self, pred_completion=None, gold_completion=None, **kwargs):


    #     return outputs
    
    # def call_by_row(self, row):




    # def call_by_dict(self, the_dict)









