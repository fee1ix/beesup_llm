from beesup_llm import get_labhandler

import torch
import logging
import numpy as np
import pandas as pd

from transformers import AutoModel, BitsAndBytesConfig

class EMBPipeline(object):
    logger = logging.getLogger(__name__)

    @classmethod
    def from_model(cls, model=None):

        name_or_path = getattr(model, 'name_or_path', None)
        if name_or_path in ['nvidia/NV-Embed-v2']:
            return NVEmbedPipeline(model=model, name_or_path=name_or_path)

        return cls(model=model, name_or_path=name_or_path)

    def __init__(self, ref=None, labh=get_labhandler(), **kwargs):

        if 'model' in kwargs:
            self.model=kwargs.pop('model')
            self.name_or_path=getattr(self.model, 'name_or_path', None)
        self.name_or_path = getattr(self, 'name_or_path', None) or  kwargs.get('name_or_path', None)

        self.instruction = ""

        if labh is not None:
            self.labh=labh
            self.labh.attach_parent(locals())

    def load_model(self):
        self.logger.info(f"Loading model {self.name_or_path}")
        self.model=AutoModel.from_pretrained(
            self.name_or_path,
            device_map='auto',
            quantization_config=BitsAndBytesConfig(
                bnb_4bit_compute_dtype=torch.bfloat16,
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            ),
            trust_remote_code=True
        )
        return

    def get_model(self):

        model=getattr(self, 'model', None)
        if model is None:
            self.load_model()
            model=self.model
            del self.model
            return model
        
        else:
            return self.model

    def prepare(self):
        if not hasattr(self, 'model'): self.load_model()
        return

    def get_emb(self, text:str, instruction:str=None, **kwargs) -> np.ndarray:
        instruction=instruction or getattr(self, 'instruction', None)
        self.prepare()
        
        emb = self.model.encode([text], instruction=instruction)[0].cpu().numpy()
        torch.cuda.empty_cache()

        return emb
    
    def get_emb_batch(self, batch: list, instruction:str=None, **kwargs) -> np.ndarray:
        instruction=instruction or getattr(self, 'instruction', None)
        self.prepare()

        emb_batch = self.model.encode(batch, instruction=instruction)
        emb_batch = [emb.cpu().numpy() for emb in emb_batch] # convert to numpy array/ move to cpu
        torch.cuda.empty_cache()

        return emb_batch

    def add_emb(self, pipe_df: pd.DataFrame, batch_size: int = 16, txt_key:str='text', emb_key:str='emb', verbose=False, **kwargs):
        assert txt_key in pipe_df.columns, f"Column '{txt_key}' not found in DataFrame"

        embs=[]
        for i in range(0, len(pipe_df), batch_size):

            text_batch = pipe_df[i:i+batch_size][txt_key].to_list()
            emb_batch = self.get_emb_batch(text_batch, **kwargs)

            embs.extend(emb_batch)

            if verbose: print(f"{i+len(emb_batch)}/{len(pipe_df)}"+25*" ", end="\r")

        pipe_df[emb_key]=embs
        return

    def call_on_dataframe(self, pipe_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.add_emb(pipe_df, **kwargs)
        return pipe_df

    def __call__(self, pipe_input, **kwargs):

        if isinstance(pipe_input, pd.DataFrame):
            return self.call_on_dataframe(pipe_input, **kwargs)
        
        # elif isinstance(pipe_input, (pd.Series, dict)):
        #     return self.call_on_sample(pipe_input, **kwargs)
    
        elif isinstance(pipe_input, list) and all([isinstance(i, str) for i in pipe_input]):
            return self.get_emb_batch(pipe_input, **kwargs)
        
        elif isinstance(pipe_input, str):
            return self.get_emb(pipe_input, **kwargs)

class NVEmbedPipeline(EMBPipeline):

    def __init__(self, ref=None, **kwargs):
        super().__init__(ref, **kwargs)

        self.name_or_path = getattr(self,'name_or_path', None) or 'nvidia/NV-Embed-v2' #https://huggingface.co/nvidia/NV-Embed-v2

