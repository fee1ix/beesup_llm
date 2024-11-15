from beesup_llm import *
from ..toolkit.setup_utils import *
from ..toolkit.llm_utils import *

from beesup_llm.dataset import BaseDataset
from beesup_llm.training import LoraTrainer
from beesup_llm.extraction.extraction_pipeline import ExtractionPipeline


class ExtractionExperiment(BaseDirectory):
    type='extraction_experiment'

    def __init__(self, ref=None, dataset_ref=None, pipe_ref=None, trainer_ref=None, **kwargs):
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