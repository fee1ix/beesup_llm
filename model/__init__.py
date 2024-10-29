from beesup_llm import *
from ..toolkit.setup_utils import *

import logging

import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig, set_seed

class BaseModelWrap(BaseDirectory):

    def __new__(cls, ref=None):

        temp_instance = super().__new__(cls)
        temp_instance.__init__(ref)

        print(temp_instance.name_or_path)

        #print(temp_instance.get_config())

        if hasattr(temp_instance, 'name_or_path'):
            print('name_or_path',temp_instance.name_or_path)
            if temp_instance.name_or_path == 'meta-llama/Meta-Llama-3.1-8B-Instruct':
                print('returning LlamaModelWrap')
                return super(BaseModelWrap,LlamaModelWrap).__new__(LlamaModelWrap)
        
        return super().__new__(cls)

    def __init__(self, ref=None):

        self.type='model'
        if isinstance(ref, torch.nn.Module):
            self.model=ref

        super().__init__(ref)

        self._config_key_order.extend([])
        self._config_keys_to_exclude.extend(['model','inference_tokenizer','training_tokenizer'])

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

        if not hasattr(self, 'model'):
            self.load_model()
            model=self.model
            self.model=None
            return model
        
        else:
            return self.model

    def load_inference_tokenizer(self):

        self.inference_tokenizer=AutoTokenizer.from_pretrained(
            self.name_or_path,
            **self.inference_tokenizer_config
        )

        return
    
    def get_inference_tokenizer(self):

        if not hasattr(self, 'inference_tokenizer'):
            self.load_inference_tokenizer()
            inference_tokenizer=self.inference_tokenizer
            self.inference_tokenizer=None
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

        if not hasattr(self, 'training_tokenizer'):
            self.load_training_tokenizer()
            training_tokenizer=self.training_tokenizer
            self.training_tokenizer=None
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

    def inference_loop(self, dataloader, return_df=False, **kwargs):

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


class LlamaModelWrap(BaseModelWrap):

    def __init__(self, ref=None):
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




