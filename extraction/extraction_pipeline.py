from beesup_llm import *
from ..toolkit.setup_utils import *

from beesup_llm.model import *
from beesup_llm.dataset import *


from .extraction_utils import *



class ExtractionPipeline(object):

    def __init__(self, dataset_ref=None, model_ref=None, **kwargs):

        if dataset_ref is not None:
            if isinstance(dataset_ref, pd.DataFrame):
                self.df=dataset_ref

            else:
                dataset=BaseDataset(dataset_ref)
                self.dataset_config=dataset.get_config()
                self.df=dataset.dataset_df

        if model_ref is not None:
            self.modelwrap=GenModelWrap.from_ref(model_ref)
            #self.gen_model_config=self.modelwrap.get_config()
            #self.tokenizer=self.modelwrap.get_inference_tokenizer()
    
    def __call__(self, report_passage, **kwargs):

        prompt_messages=get_prompt_messages(report_passage, use_extraction_prompt=True, use_few_shots=True)
        outputs=self.modelwrap(prompt_messages,**kwargs)

        # #prompt_ids=self.get_prompt_ids(report_passage, **kwargs)

        # outputs=self.modelwrap.inference_step({'input':prompt_ids})

        return outputs
    
    



