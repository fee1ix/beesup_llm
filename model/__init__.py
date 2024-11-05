from beesup_llm import *
from ..toolkit.setup_utils import *

import logging

import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig, set_seed

class BaseModelWrap(BaseDirectory):

    type='model'

    @staticmethod
    def get_subconfig(ref):

        gen_model=getattr_or_key(ref,'gen_model_config',False)
        emb_model=getattr_or_key(ref,'emb_model_config',False)

        if gen_model and emb_model: raise ValueError("Both gen_model and emb_model are present in the config.")
        if gen_model: return gen_model
        if emb_model: return emb_model
        else: return None
            
    @staticmethod
    def is_GenModelWrap(ref):
        if getattr_or_key(ref, 'type') == 'gen_model': return True

        if getattr_or_key(ref, 'name_or_path') in [
            'meta-llama/Meta-Llama-3.1-8B-Instruct',
            'mistralai/Mistral-7B-Instruct-v0.2',
            ]: return True

        return False
    
    @staticmethod
    def is_EmbModelWrap(ref):
        if getattr_or_key(ref, 'type') == 'emb_model': return True

        if getattr_or_key(ref, 'name_or_path') in [
            'jinaai/jina-embeddings-v3',
            ]: return True

        return False

    
    def __new__(cls, ref=None, skip_new=False, **kwargs):
        kwargs.update(get_cls_attrs(cls))
        if skip_new: return super().__new__(cls)
        if cls is not BaseModelWrap: return super().__new__(cls)
        if cls.get_subconfig(ref) is not None: return BaseModelWrap(cls.get_subconfig(ref), skip_new=True, **kwargs)

        pre_config=get_config_from_ref(ref,**kwargs)
        
        if cls.is_GenModelWrap(pre_config): return GenModelWrap(pre_config, **kwargs)
        if cls.is_EmbModelWrap(pre_config): return EmbModelWrap(pre_config, **kwargs)

        return BaseModelWrap(pre_config, skip_new=True, **kwargs)
    
    def __init__(self, ref=None, **kwargs):

        super().__init__(ref, **kwargs)
        self._config_key_order.extend(['name_or_path'])
        self._config_keys_to_exclude.extend(['model'])

        self._default_config=dict()
        self.update_attributes(self._default_config, overwrite=False)
    
    def __preinit__from_model(self, model):
        self.name_or_path=model.name_or_path
        self.model=model



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
    
    def get_model(self):

        model=getattr(self, 'model', None)
        if model is None:
            self.load_model()
            model=self.model
            del self.model
            return model
        
        else:
            return self.model


# class BaseLlmWrap(BaseModelWrap):

# class BaseEmbWrap(BaseModelWrap):

class GenModelWrap(BaseModelWrap):

    type='gen_model'
  
    @staticmethod
    def is_LlamaModelWrap(ref):
        if getattr_or_key(ref, 'name_or_path') == 'meta-llama/Meta-Llama-3.1-8B-Instruct': return True
        return False

    def __new__(cls, ref=None, skip_new=False, **kwargs):
        kwargs.update(get_cls_attrs(cls))
        print(kwargs)

        if skip_new: return super().__new__(cls)
        if cls is not GenModelWrap: return super().__new__(cls)

        pre_config=get_config_from_ref(ref,**kwargs)
        
        if cls.is_LlamaModelWrap(pre_config): return LlamaModelWrap(pre_config)

        return GenModelWrap(pre_config, skip_new=True, **kwargs)


    def __init__(self, ref=None, **kwargs):

        super().__init__(ref, **kwargs)
        self._config_key_order.extend([])
        self._config_keys_to_exclude.extend(['inference_tokenizer','training_tokenizer'])

        self._default_config=dict(
            bnb_config=dict(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            ),
            inference_tokenizer_config=dict(
                padding_side='left',
                padding='longest',
                add_special_tokens=True,
            ),
            training_tokenizer_config=dict(
                padding_side='right',
                padding='longest',
                add_special_tokens=True,
            ),
            generation_config=dict(
                return_dict_in_generate=True,
                output_scores=False,
                output_logits=False,
                max_new_tokens=750,
                max_time=600,
                stop_strings=None,
            )
        )
        
        self.update_attributes(self._default_config, overwrite=False)

    def load_inference_tokenizer(self):

        self.inference_tokenizer=AutoTokenizer.from_pretrained(
            self.name_or_path,
            **self.inference_tokenizer_config
        )

        return
    
    def get_inference_tokenizer(self):

        inference_tokenizer=getattr(self, 'inference_tokenizer', None)
        if inference_tokenizer is None:
            self.load_inference_tokenizer()
            inference_tokenizer=self.inference_tokenizer
            del self.inference_tokenizer
            return inference_tokenizer
        
        else:
            return self.inference_tokenizer

    def load_training_tokenizer(self):

        self.training_tokenizer=AutoTokenizer.from_pretrained(
            self.name_or_path,
            **self.training_tokenizer_config
        )

        return

    def get_training_tokenizer(self):

        training_tokenizer=getattr(self, 'training_tokenizer', None)
        if training_tokenizer is None:
            self.load_training_tokenizer()
            training_tokenizer=self.training_tokenizer
            del self.training_tokenizer
            return training_tokenizer
        
        else:
            return self.training_tokenizer
            
    def inference_step(self,inputs,**kwargs):

        generation_config=GenerationConfig.from_dict(self.generation_config)
    
        if kwargs.get('seed'): set_seed(kwargs.get('seed'))

        if kwargs: generation_config.update(**kwargs)
        if kwargs.get('generation_config'): generation_config.update(**kwargs.get('generation_config'))
        
        inputs.to("cuda")
        outputs=self.model.generate(
            generation_config=generation_config,
            tokenizer=self.inference_tokenizer,
            **inputs)
        
        return outputs

    def inference_loop(self, dataloader, **kwargs):

        import torch
        from transformers.trainer_pt_utils import EvalLoopContainer

        # Initialize containers
        all_input_ids = EvalLoopContainer(do_nested_concat=True, padding_index=-100)
        all_label_ids = EvalLoopContainer(do_nested_concat=True, padding_index=-100)
        all_pred_ids = EvalLoopContainer(do_nested_concat=True, padding_index=-100)
        all_all_ids = EvalLoopContainer(do_nested_concat=True, padding_index=-100)
        all_losses = EvalLoopContainer(do_nested_concat=True, padding_index=-100)

        for step, inputs in enumerate(dataloader):

            input_ids, label_ids, pred_ids,all_ids, losses  = None, None, None, None, None
            
            input_ids=inputs.get('input_ids',None)
            label_ids=inputs.get('labels',None)

            self.logger.info(f'step = {step}')

            all_ids=self.inference_step(inputs, **kwargs)['sequences']

     
            if input_ids is not None: all_input_ids.add(input_ids)
            if label_ids is not None: all_label_ids.add(label_ids)

            if all_ids is not None: all_all_ids.add(all_ids)

            if pred_ids is not None: all_pred_ids.add(pred_ids)
            if losses is not None: all_losses.add(losses)

            del input_ids, label_ids, pred_ids, losses, all_ids
            torch.cuda.empty_cache()

        return  { 
            'all_input_ids': all_input_ids.get_arrays(),
            'all_label_ids': all_label_ids.get_arrays(),
            'all_pred_ids': all_pred_ids.get_arrays(),
            'all_all_ids': all_all_ids.get_arrays(),
            'all_losses': all_losses.get_arrays(),
        }



class LlamaModelWrap(GenModelWrap):
    type='gen_model'

    @staticmethod
    def is_PeftLlamaModelWrap(ref):
        return False

    def __new__(cls, ref=None, skip_new=False, **kwargs):
        kwargs.update(get_cls_attrs(cls))
        print(kwargs)

        if skip_new: return super().__new__(cls)
        if cls is not LlamaModelWrap: return super().__new__(cls)

        pre_config=get_config_from_ref(ref,**kwargs)
        
        if cls.is_PeftLlamaModelWrap(pre_config): return PeftLlamaModelWrap(pre_config)

        return LlamaModelWrap(pre_config, skip_new=True, **kwargs)

    def __init__(self, ref=None, **kwargs):
        super().__init__(ref)

        self._config_key_order.extend(['name_or_path','base_model'])
        self._config_keys_to_exclude.extend([])

        self._default_config=dict(
            name_or_path='meta-llama/Meta-Llama-3.1-8B-Instruct',
            base_model='Meta-Llama-3.1-8B-Instruct',

            inference_tokenizer_config=dict(
                max_length=8192,
                pad_token='<|begin_of_text|>',
                pad_token_id=128000
            ),
            training_tokenizer_config=dict(
                max_length=8192,
                pad_token='<|end_of_text|>',
                pad_token_id=128001
            ),
            generation_config=dict(
                pad_token='<|begin_of_text|>',
                pad_token_id=128000,
            )
        )
        
        self.update_attributes(self._default_config, overwrite=False)

from peft import AutoPeftModelForCausalLM
class PeftLlamaModelWrap(LlamaModelWrap):

    def __init__(self, ref=None, **kwargs):
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



class EmbModelWrap(BaseModelWrap):
    type='emb_model'


# class JinaiModelWrap(BaseModelWrap):

#     def __init__(self, ref=None, **kwargs):
#         super().__init__(ref)

#         self._config_key_order.extend([])
#         self._config_keys_to_exclude.extend([])






