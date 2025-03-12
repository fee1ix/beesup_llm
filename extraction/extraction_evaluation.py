from beesup_llm import get_labhandler, _isinstance
from beesup_llm.llm_evaluation import *

from beesup_llm.extraction.extraction_pipeline import ExtractionPipeline


class ExtractionEvaluator(LLMEvaluator):

    def __init__(self, *args, extraction_pipe: ExtractionPipeline, **kwargs):
        super().__init__(*args, **kwargs)

        if hasattr(self, 'labh'):
            extraction_pipe = self.labh.handle_object(locals(), 'extraction_pipe')

        if _isinstance(extraction_pipe, ExtractionPipeline):
            self.extraction_pipe=extraction_pipe
            self.extraction_pipe.llm_pipe=getattr(self, 'llm_pipe', None) or getattr(self.extraction_pipe, 'llm_pipe', None)
            
            if not hasattr(self.extraction_pipe, 'llm_pipe'):
                raise ValueError("neighter llm_pipe in args nor in extraction_pipe")
        
        
    





