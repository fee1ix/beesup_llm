from beesup_llm import *
from beesup_llm.toolkit.llm_utils import *
from beesup_llm.toolkit.setup_utils import *

import torch
import logging
import pandas as pd



class BaseModelPipeline(BaseDirectory):
    type='model_pipeline'

    @staticmethod
    def matches(ref):
        if getattr_or_key(ref, 'type') == ['model']: return True
        if LanguageModelPipeline.matches(ref): return True
        if EmbeddingModelPipeline.matches(ref): return True
        return False

    @staticmethod
    def get_subconfig(ref):

        llm_pipeline=getattr_or_key(ref,f'{LanguageModelPipeline.type}_config',False)
        emb_pipeline=getattr_or_key(ref,f'{EmbeddingModelPipeline.type}_config',False)

        if llm_pipeline and emb_pipeline: raise ValueError("Both gen_model and emb_model are present in the config.")
        if llm_pipeline: return llm_pipeline
        if emb_pipeline: return emb_pipeline
        else: return None
    
    @classmethod
    def from_ref(cls, ref=None, **kwargs):
        kwargs.update(get_cls_attrs(cls))
        cls.logger.debug(f"{cls} ref={ref}, kwargs = {kwargs}\n")

        if isinstance(ref, cls): return ref

        pre_config = get_config_from_ref(ref, **kwargs)

        if LanguageModelPipeline.matches(pre_config):
            return LanguageModelPipeline.from_ref(ref=pre_config, **kwargs)
        
        if EmbeddingModelPipeline.matches(pre_config):
            return EmbeddingModelPipeline.from_ref(ref=pre_config, **kwargs)

        return cls(ref=pre_config, **kwargs)

    def __init__(self, ref=None, **kwargs):
        self.logger.debug(f"{self.__class__} ref={ref}, kwargs = {kwargs}\n")
        super().__init__(ref, **kwargs)

        self._default_config=dict(     
        )
        self._config_key_order.extend(list(self._default_config.keys())+['name_or_path'])
        self._config_keys_to_exclude.extend(['model'])

        if 'model' in kwargs:
            self.model = kwargs.pop('model')
        
        self.update_config(self._default_config, overwrite_if_conflict=False)
        self.update_config_smart(
            kwargs, 
            interpret_none_as_val=True, 
            overwrite_if_conflict=True, 
            allow_new_atomic_keys=False, 
            allow_new_nested_keys=False
        )


    def get_tokenizer(self,**kwargs):

        tokenizer=AutoTokenizer.from_pretrained(
            self.name_or_path,
            **kwargs
        )

        return tokenizer

        



    def get_model(self):

        model=getattr(self, 'model', None)
        if model is None:
            self.load_model()
            model=self.model
            del self.model
            return model
        
        else:
            return self.model


from threading import Thread
from transformers import \
    AutoTokenizer, \
    AutoModelForCausalLM, \
    BitsAndBytesConfig, \
    GenerationConfig, \
    set_seed, \
    TextGenerationPipeline, \
    TextIteratorStreamer


logging.getLogger("transformers").setLevel(logging.ERROR)
    
class LanguageModelPipeline(BaseModelPipeline):
    type='llm_pipeline'

    @staticmethod
    def matches(ref):
        if LlamaPipeline.matches(ref): return True
        if MistralPipeline.matches(ref): return True
        return False

    @classmethod
    def from_ref(cls, ref=None, **kwargs):
        kwargs.update(get_cls_attrs(cls))
        cls.logger.debug(f"{cls} ref={ref}, kwargs = {kwargs}\n")

        pre_config = get_config_from_ref(ref, **kwargs)

        if getattr(ref, 'type', None) == cls.type:
            if hasattr(ref,'model'): kwargs['model']=ref.model
            if hasattr(ref,'inference_tokenizer'): kwargs['inference_tokenizer']=ref.inference_tokenizer
            if hasattr(ref,'training_tokenizer'): kwargs['training_tokenizer']=ref.training_tokenizer
        
        if isinstance(ref, torch.nn.Module):
            kwargs['model']=ref

        if LlamaPipeline.matches(pre_config): return LlamaPipeline(ref=pre_config, **kwargs)
        if MistralPipeline.matches(pre_config): return MistralPipeline(ref=pre_config, **kwargs)
        if MistralNemoPipeline.matches(pre_config): return MistralNemoPipeline(ref=pre_config, **kwargs)
        
        return cls(ref=pre_config, **kwargs)

    def __init__(self, ref=None, **kwargs):
        self.logger.debug(f"{self.__class__} ref={ref}, kwargs = {kwargs}\n")
        super().__init__(ref, **kwargs)

        self._default_config=dict(
            pipeline_args=dict(
                return_full_text=False,
                clean_up_tokenization_spaces=True,
            ),
            bnb_config=dict(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            ),
            inference_tokenizer_config=dict(
                padding_side='left',
                padding='longest',
                #add_special_tokens=True,
            ),
            training_tokenizer_config=dict(
                padding_side='right',
                padding='longest',
                #add_special_tokens=True,
            ),
            generation_config=dict(
                return_dict_in_generate=False,
                max_new_tokens=750,
                max_time=600,
                stop_strings=None,
            ),
        )

        self._config_key_order.extend(list(self._default_config.keys()))
        self._config_keys_to_exclude.extend(['inference_tokenizer','training_tokenizer','outputs'])
        
        self.update_config(self._default_config, overwrite_if_conflict=False)

        self.update_config_smart(
            kwargs, 
            interpret_none_as_val=True, 
            overwrite_if_conflict=True, 
            allow_new_atomic_keys=False, 
            allow_new_nested_keys=False
        )

    def load_model(self):

        self.logger.info(f"Loading model {self.name_or_path}")
        self.model=AutoModelForCausalLM.from_pretrained(
            self.name_or_path,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(
                bnb_4bit_compute_dtype=torch.bfloat16,
                **self.bnb_config
                ),
        )

        return
    
    def load_tokenizer(self, tokenizer_type='inference'):
                
        setattr(self, f'{tokenizer_type}_tokenizer', AutoTokenizer.from_pretrained(
            self.name_or_path,
            **getattr(self, f'{tokenizer_type}_tokenizer_config')
        ))

        return
    
    def get_tokenizer(self, tokenizer_type='inference'):

        tokenizer=getattr(self, f'{tokenizer_type}_tokenizer', None)
        if tokenizer is None:
            self.load_tokenizer(tokenizer_type)
            tokenizer=getattr(self, f'{tokenizer_type}_tokenizer')
            return tokenizer
        
        else:
            return tokenizer
    
    def load_inference_tokenizer(self):

        self.inference_tokenizer=AutoTokenizer.from_pretrained(
            self.name_or_path,
            **self.inference_tokenizer_config
        )

        return
    
    def get_inference_tokenizer(self, keep=False):

        inference_tokenizer=getattr(self, 'inference_tokenizer', None)
        if inference_tokenizer is None:
            self.load_inference_tokenizer()
            inference_tokenizer=self.inference_tokenizer
            if not keep:
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

    def count_tokens(self, pipe_input):
        tokenizer=self.get_inference_tokenizer()
        return sum(tokenizer(pipe_input, return_length=True)['length'])

    def load_pipeline(self,**kwargs):
        self.pipeline = TextGenerationPipeline(model=self.model, tokenizer=self.inference_tokenizer, **kwargs)

    def get_pipeline(self, **kwargs):
        pipeline=getattr(self, 'pipeline', None)
        if pipeline is None:
            self.load_pipeline(**kwargs)
            pipeline=self.pipeline
            del self.pipeline
            return pipeline
        
        else:
            return self.pipeline

    def prepare_inference(self):

        if not hasattr(self, 'model'): self.load_model()
        if not hasattr(self, 'inference_tokenizer'): self.load_tokenizer('inference')
        if not hasattr(self, 'pipeline'): self.load_pipeline()
    
    def yield_completion_stream(self, pipe_input, **kwargs):

        self.prepare_inference()
        streamer = TextIteratorStreamer(self.inference_tokenizer, skip_prompt=True)

        self.load_pipeline(streamer=streamer)

        generation_config=self.get_updated_config(kwargs, config_key='generation_config')
        self._recent_generation_config=generation_config

        pipeline_kwargs={
            'text_inputs':pipe_input,
            'tokenizer':self.pipeline.tokenizer,
            'generation_config':GenerationConfig.from_dict(generation_config),
            **self.get_updated_config(kwargs, config_key='pipeline_args')
        }
        
        streamer_thread=Thread(target=self.pipeline, kwargs=pipeline_kwargs)

        try:
            streamer_thread.start()
            for new_token in streamer:
                if new_token in self.pipeline.tokenizer.special_tokens_map.values(): continue
                yield new_token

            streamer_thread.join()
        
        except Exception as e:
            self.logger.info(f'{e}')
            streamer_thread.join()
            torch.cuda.empty_cache()
            self.logger.info(f'generation_stream: executed torch.cuda.empty_cache()')
        
        del streamer
        del self.pipeline
        return
    
    def print_completion_stream(self, pipe_input, **kwargs):

        completion=""
        for new_token in self.yield_completion_stream(pipe_input, **kwargs):
            print(new_token, end='', flush=True)
            completion+=new_token
        
        return completion

    def get_output(self, pipe_input, **kwargs):

        self.logger.debug(f"pipe_input: {pipe_input}, kwargs: {kwargs}")

        generation_config=self.get_updated_config(kwargs, config_key='generation_config')
        self._recent_generation_config=generation_config

        self.logger.debug(f"{self._recent_generation_config}")

        
        self.prepare_inference()

        pipeline_kwargs={
            'text_inputs':pipe_input,
            'tokenizer':self.pipeline.tokenizer,
            'generation_config':GenerationConfig.from_dict(generation_config),
            **self.get_updated_config(kwargs, config_key='pipeline_args')
        }
        return self.pipeline(**pipeline_kwargs)
    
    def get_pred_completion(self, pipe_input, **kwargs):
        return self.get_output(pipe_input, **kwargs)[0]['generated_text']
    
    def add_pred_completion(self, pipe_df: pd.DataFrame, **kwargs):
        
        if 'prompt_messages' in pipe_df.columns:
            pipe_input=list(pipe_df['prompt_messages'].values)
        
        elif 'prompt' in pipe_df.columns:
            pipe_input=list(pipe_df['prompt'].values)

        the_output=self.get_output(pipe_input, **kwargs)

        pipe_df['pred_completion']=[o[0]['generated_text'] for o in the_output]

    def call_on_dataframe(self, pipe_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.add_pred_completion(pipe_df, **kwargs)
        return pipe_df

    def call_on_single(self, pipe_input: Union[str, list], use_chatformat=False, stream=False, **kwargs) -> str:

        if use_chatformat:
            if isinstance(pipe_input, str):
                pipe_input=[{'role':'user','content':pipe_input}]
            
            elif isinstance(pipe_input, list) and all(isinstance(x, str) for x in pipe_input):
                pipe_input=[[{'role':'user','content':x}] for x in pipe_input]
        
        if stream:
            return self.print_completion_stream(pipe_input, **kwargs)
        else:
            return self.get_pred_completion(pipe_input, **kwargs)

    def call_on_sample(self, sample: Union[pd.Series, dict], **kwargs) -> str:

        if 'prompt_messages' in sample:
            pipe_input=sample['prompt_messages']
        elif 'prompt' in sample:
            pipe_input=sample['prompt']
        
        return self.call_on_single(pipe_input, **kwargs)

    def __call__(self, pipe_input, stream=False, use_chatformat=False, **kwargs):

        if isinstance(pipe_input, pd.DataFrame):
            return self.call_on_dataframe(pipe_input, **kwargs)
        
        elif isinstance(pipe_input, (pd.Series, dict)):
            return self.call_on_sample(pipe_input, **kwargs)
        
        else:
            return self.call_on_single(pipe_input, use_chatformat=use_chatformat, stream=stream, **kwargs)
        

        
class LlamaPipeline(LanguageModelPipeline):

    @staticmethod
    def matches(ref):
        if getattr_or_key(ref, 'name_or_path') == 'meta-llama/Meta-Llama-3.1-8B-Instruct': return True
        return False

    def __init__(self, ref=None, **kwargs):
        self.logger.debug(f"{self.__class__} ref={ref}, kwargs = {kwargs}\n")
        super().__init__(ref, **kwargs)

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
                stop_strings=None,
                pad_token_id=128000,
            )
        )

        self._config_key_order.extend(list(self._default_config.keys()))
        self._config_keys_to_exclude.extend([])
        
        self.update_config(self._default_config, overwrite_if_conflict=True)
        self.update_config_smart(
            kwargs, 
            interpret_none_as_val=True, 
            overwrite_if_conflict=True, 
            allow_new_atomic_keys=False, 
            allow_new_nested_keys=False
        )

class MistralPipeline(LanguageModelPipeline):

    @staticmethod
    def matches(ref):
        if getattr_or_key(ref, 'name_or_path') in ['mistralai/Mistral-7B-Instruct-v0.2']: return True
        return False

    def __init__(self, ref=None, **kwargs):
        self.logger.debug(f"{self.__class__} ref={ref}, kwargs = {kwargs}\n")
        super().__init__(ref, **kwargs)

        self._default_config=dict(
            name_or_path='mistralai/Mistral-7B-Instruct-v0.2',
            base_model='Mistral-7B-Instruct-v0.2',

            inference_tokenizer_config=dict(
                max_length=8192,
                pad_token='<unk>',
            ),
            training_tokenizer_config=dict(
                max_length=8192,
                pad_token='<unk>',
            ),
            generation_config=dict(
                pad_token='<unk>',
            )
        )

        self._config_key_order.extend(list(self._default_config.keys()))
        self._config_keys_to_exclude.extend([])
        
        self.update_config(self._default_config, overwrite_if_conflict=True)
        self.update_config_smart(
            kwargs, 
            interpret_none_as_val=True, 
            overwrite_if_conflict=True, 
            allow_new_atomic_keys=False, 
            allow_new_nested_keys=False
        )

class MistralNemoPipeline(LanguageModelPipeline):

    @staticmethod
    def matches(ref):
        if getattr_or_key(ref, 'name_or_path') in ["mistralai/Mistral-Nemo-Instruct-2407"]: return True
        return False

    def __init__(self, ref=None, **kwargs):
        self.logger.debug(f"{self.__class__} ref={ref}, kwargs = {kwargs}\n")
        super().__init__(ref, **kwargs)

        self._default_config=dict(
            name_or_path='mistralai/Mistral-Nemo-Instruct-2407',
            base_model='Mistral-Nemo-Instruct-2407',

            inference_tokenizer_config=dict(
                max_length=8192,
                pad_token='<s>',
            ),
            training_tokenizer_config=dict(
                max_length=8192,
                pad_token='</s>',
            ),
            generation_config=dict(
                pad_token='<s>',
            )
        )

        self._config_key_order.extend(list(self._default_config.keys()))
        self._config_keys_to_exclude.extend([])
        
        self.update_config(self._default_config, overwrite_if_conflict=True)
        self.update_config_smart(
            kwargs, 
            interpret_none_as_val=True, 
            overwrite_if_conflict=True, 
            allow_new_atomic_keys=False, 
            allow_new_nested_keys=False
        )

class PhiPipeline(LanguageModelPipeline):

    @staticmethod
    def matches(ref):
        if getattr_or_key(ref, 'name_or_path') in ["microsoft/Phi-3-medium-128k-instruct"]: return True
        return False

    def __init__(self, ref=None, **kwargs):
        self.logger.debug(f"{self.__class__} ref={ref}, kwargs = {kwargs}\n")
        super().__init__(ref, **kwargs)

        self._default_config=dict(
            name_or_path='microsoft/Phi-3-medium-128k-instruct',
            base_model='Phi-3-medium-128k-instruct',

            inference_tokenizer_config=dict(
                max_length=8192,
                pad_token='<s>',
            ),
            training_tokenizer_config=dict(
                max_length=8192,
                pad_token='<|endoftext|>',
            ),
            generation_config=dict(
                pad_token='<s>',
            )
        )

        self._config_key_order.extend(list(self._default_config.keys()))
        self._config_keys_to_exclude.extend([])
        
        self.update_config(self._default_config, overwrite_if_conflict=True)
        self.update_config_smart(
            kwargs, 
            interpret_none_as_val=True, 
            overwrite_if_conflict=True, 
            allow_new_atomic_keys=False, 
            allow_new_nested_keys=False
        )


from transformers import AutoModel

class EmbeddingModelPipeline(BaseModelPipeline):
    type='emb_pipeline'

    @staticmethod
    def matches(ref):
        if NVEmbedPipeline.matches(ref): return True
        return False

    @classmethod
    def from_ref(cls, ref=None, **kwargs):
        kwargs.update(get_cls_attrs(cls))
        cls.logger.debug(f"{cls} ref={ref}, kwargs = {kwargs}\n")

        pre_config = get_config_from_ref(ref, **kwargs)

        if getattr(ref, 'type', None) == cls.type:
            if hasattr(ref,'model'): kwargs['model']=ref.model
        
        if isinstance(ref, torch.nn.Module):
            kwargs['model']=ref

        if NVEmbedPipeline.matches(pre_config): return NVEmbedPipeline(ref=pre_config, **kwargs)
        return cls(ref=pre_config, **kwargs)

    def __init__(self, ref=None, **kwargs):
        self.logger.debug(f"{self.__class__} ref={ref}, kwargs = {kwargs}\n")
        super().__init__(ref, **kwargs)

        self._default_config=dict(
            bnb_config=dict(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            ),
            model_load_config=dict(
                trust_remote_code=True,
            ),
            tokenizer_config=dict(
            ),
            encode_config=dict(  
            ),
        )

        self._config_key_order.extend(list(self._default_config.keys()))
        self._config_keys_to_exclude.extend([])
        
        self.update_config(self._default_config, overwrite_if_conflict=False)
        self.update_config_smart(
            kwargs, 
            interpret_none_as_val=True, 
            overwrite_if_conflict=True, 
            allow_new_atomic_keys=False, 
            allow_new_nested_keys=False
        )

    def load_tokenizer(self):
        self.tokenizer=self.get_tokenizer()

    def load_model(self):

        self.logger.info(f"Loading model {self.name_or_path}")
        self.model = AutoModel.from_pretrained(
            self.name_or_path,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(
                bnb_4bit_compute_dtype=torch.bfloat16,
                **self.bnb_config
                ),
            **self.model_load_config
            )

        return


class NVEmbedPipeline(EmbeddingModelPipeline):

    @staticmethod
    def matches(ref):
        if getattr_or_key(ref, 'name_or_path') == 'nvidia/NV-Embed-v2': return True
        return False

    def __init__(self, ref=None, **kwargs):
        self.logger.debug(f"{self.__class__} ref={ref}, kwargs = {kwargs}\n")
        super().__init__(ref, **kwargs)

        self._default_config=dict(
            name_or_path='nvidia/NV-Embed-v2',
            base_model='NV-Embed-v2',

        )

        self._config_key_order.extend(list(self._default_config.keys()))
        self._config_keys_to_exclude.extend([])
        
        self.update_config(self._default_config, overwrite_if_conflict=True)
        self.update_config_smart(
            kwargs, 
            interpret_none_as_val=True, 
            overwrite_if_conflict=True, 
            allow_new_atomic_keys=False, 
            allow_new_nested_keys=False
        )

    
