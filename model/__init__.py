from beesup_llm import *
from ..toolkit.setup_utils import *

import logging

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


#def parameter_update(default_params, new_params):


class BaseModelWrap(BaseDirectory):

    def __init__(self, ref=None):

        self.type='model'
        super().__init__(ref)

        self._config_key_order.extend([])
        self._config_keys_to_exclude.extend(['model','inference_tokenizer','training_tokenizer'])

        bnb_config=dict(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            )
        if hasattr(self,'bnb_config'):
            bnb_config.update(self.bnb_config)
            self.bnb_config=bnb_config
        else:  self.bnb_config=bnb_config

        inference_tokenizer_config=dict(
            padding_side='left',
            padding='longest',
            add_special_tokens=True,
        )
        if hasattr(self,'inference_tokenizer_config'):
            inference_tokenizer_config.update(self.inference_tokenizer_config)
            self.inference_tokenizer_config=inference_tokenizer_config
        else:  self.inference_tokenizer_config=inference_tokenizer_config
        
        training_tokenizer_config=dict(
            padding_side='right',
            padding='longest',
            add_special_tokens=True,
            )
        if hasattr(self,'training_tokenizer_config'):
            training_tokenizer_config.update(self.training_tokenizer_config)
            self.training_tokenizer_config=training_tokenizer_config
        else:  self.training_tokenizer_config=training_tokenizer_config

        generation_config=dict(
            return_dict_in_generate=True,
            output_scores=False,
            output_logits=False,
            max_new_tokens=750,
            max_time=600,
            stop_strings=None,
        )

        if hasattr(self,'generation_config'):
            generation_config.update(self.generation_config)
            self.generation_config=generation_config
        else:  self.generation_config=generation_config

    def load_model(self):

        self.model=AutoModelForCausalLM.from_pretrained(
            self.name_or_path,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(
                bnb_4bit_compute_dtype=torch.bfloat16,
                **self.bnb_config
                ),
        )

        return
    
    def load_inference_tokenizer(self):

        self.inference_tokenizer=AutoTokenizer.from_pretrained(
            self.name_or_path,
            **self.inference_tokenizer_config
        )

        return
    
    def load_training_tokenizer(self):

        self.training_tokenizer=AutoTokenizer.from_pretrained(
            self.name_or_path,
            **self.training_tokenizer_config
        )

        return


    def spawn(self):

        if not os.path.exists(f'{self.path}'):
            os.makedirs(f'{self.path}', exist_ok=False)

        set_config(self.get_config())
        logging.info(f"{self.name.upper()} spawned at {self.path}")


class LlamaModelWrap(BaseModelWrap):

    def __init__(self, ref=None):
        super().__init__(ref)

        self._config_key_order.extend(['name_or_path'])
        self._config_keys_to_exclude.extend([])

        self.name_or_path = 'meta-llama/Meta-Llama-3.1-8B-Instruct'

        inference_tokenizer_config=dict(
            max_length=8192,
            pad_token='<|begin_of_text|>',
            pad_token_id=128000
        )
        if hasattr(self,'inference_tokenizer_config'):
            inference_tokenizer_config.update(self.inference_tokenizer_config)
            self.inference_tokenizer_config=inference_tokenizer_config
        else:  self.inference_tokenizer_config=inference_tokenizer_config
        
        training_tokenizer_config=dict(
            max_length=8192,
            pad_token='<|end_of_text|>',
            pad_token_id=128001
            )
        if hasattr(self,'training_tokenizer_config'):
            training_tokenizer_config.update(self.training_tokenizer_config)
            self.training_tokenizer_config=training_tokenizer_config
        else:  self.training_tokenizer_config=training_tokenizer_config

        generation_config=dict(
            pad_token='<|begin_of_text|>',
            pad_token_id=128000,
        )
        if hasattr(self,'generation_config'):
            generation_config.update(self.generation_config)
            self.generation_config=generation_config
        else:  self.generation_config=generation_config

    


class PeftLlamaModelWrap(LlamaModelWrap):

    def __init__(self, ref=None):
        super().__init__(ref)


    
    def load_model(self):

        self.model=AutoPeftModelForCausalLM.from_pretrained(
            self.name_or_path,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(
                bnb_4bit_compute_dtype=torch.bfloat16,
                **self.bnb_config
                ),
        )

        return




