
from beesup_llm import *
from ..toolkit.setup_utils import *

class BaseDataset(BaseDirectory):

    def __init__(self, config=None, parent_object=None, dataset_df=None):


        if config is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config = os.path.join(current_dir, 'base_dataset_config.yaml')
        
        super().__init__(config, parent_object)



    #def spawn(self):












