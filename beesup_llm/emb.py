from beesup_llm import get_labhandler

import torch

import logging
import numpy as np
import pandas as pd

from typing import List, Union

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

class EMBPipeline:
    """
    EMBPipeline is a class designed to handle embedding generation using a specified model. 
    It provides methods for loading models, generating embeddings for single texts, batches, 
    and unique texts, and integrating embeddings into pandas DataFrames.
    """
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

        self.instruction = kwargs.get('instruction', '')

        if labh is not None:
            self.labh=labh(locals())

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

    def get_embs(self, texts:list, batch_size: int = 16, instruction:str=None, verbose=False, **kwargs) -> np.ndarray:

        embs=[]
        for i in range(0, len(texts), batch_size):
            text_batch = texts[i:i+batch_size]
            emb_batch = self.get_emb_batch(text_batch, **kwargs)
            embs.extend(emb_batch)

            if verbose: print(f"{i+len(emb_batch)}/{len(texts)}"+25*" ", end="\r")
        
        return embs

    def get_embs_unique(self, texts:list, **kwargs) -> np.ndarray:

        if len(texts)!=len(set(texts)):
            txts_unique=list(set(texts))
            embs_unique=self.get_embs(txts_unique, **kwargs)
        else:
            return self.get_embs(texts, **kwargs)
        
        txt_emb_dict=dict(zip(txts_unique, embs_unique))
        embs=[txt_emb_dict[txt] for txt in texts]
        return embs

    def add_emb(self, pipe_df: pd.DataFrame, txt_key:str='text', emb_key:str='emb', **kwargs):
        assert txt_key in pipe_df.columns, f"Column '{txt_key}' not found in DataFrame"

        pipe_df[emb_key]=self.get_embs_unique(pipe_df[txt_key].to_list(), **kwargs)

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



import torch.nn.functional as F
from torch import Tensor
def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

class QwenPipeline(EMBPipeline):

    def __init__(self, ref=None, **kwargs):
        super().__init__(ref, **kwargs)

        self.name_or_path = getattr(self,'name_or_path', None) or 'Alibaba-NLP/gte-Qwen2-7B-instruct' #https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct

    def load_tokenizer(self):
        self.logger.info(f"Loading tokenizer {self.name_or_path}")
        self.tokenizer=AutoTokenizer.from_pretrained(self.name_or_path, trust_remote_code=True)
        return
    
    def get_tokenizer(self):

        tokenizer=getattr(self, 'tokenizer', None)
        if tokenizer is None:
            self.load_tokenizer()
            tokenizer=self.tokenizer
            del self.tokenizer
            return tokenizer
        
        else:
            return self.tokenizer





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
    
    def prepare(self):
        super().prepare()
        if not hasattr(self, 'tokenizer'): self.load_tokenizer()
        return

    def get_emb_batch(self, batch: List[str], instruction:str=None, **kwargs) -> np.ndarray:
        instruction=instruction or getattr(self, 'instruction', None)
        self.prepare()

        batch=[f"{instruction}\n{txt}" for txt in batch if instruction is not None]

        # Tokenize the input texts
        batch_dict = self.tokenizer(batch, max_length=8192, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(self.model.device) for k, v in batch_dict.items()}
        outputs = self.model(**batch_dict)
        emb_batch = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # normalize embeddings
        emb_batch = F.normalize(emb_batch, p=2, dim=1)
        emb_batch=emb_batch.detach().cpu().numpy()
        torch.cuda.empty_cache()
        return emb_batch

    def get_emb(self, text:str,**kwargs) -> np.ndarray:
        emb=self.get_emb_batch([text], **kwargs)[0]
        torch.cuda.empty_cache()
        return emb



    