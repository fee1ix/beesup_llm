from beesup_llm import *
from beesup_llm.toolkit.setup_utils import *
from beesup_llm.injection.evaluation_utils import *

from beesup_llm.dataset import *
from beesup_llm.model_pipelines import *

import pickle
import pandas as pd


class EvaluationSample(object):

    





class EvaluationPipeline(BaseDirectory):
    type='evaluation_pipeline'

    def __init__(self, ref=None, **kwargs):
        super().__init__(ref, **kwargs)

        self._default_config