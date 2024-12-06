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


from threading import Thread
from transformers import \
    AutoTokenizer, \
    AutoModelForCausalLM, \
    BitsAndBytesConfig, \
    GenerationConfig, \
    set_seed, \
    TextGenerationPipeline, \
    TextIteratorStreamer
    

class LanguageModelPipeline(BaseModelPipeline):
    type='llm_pipeline'

    @staticmethod
    def matches(ref):
        if getattr_or_key(ref, 'type') == 'gen_model': return True

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
        
        return cls(ref=pre_config, **kwargs)

    def __init__(self, ref=None, **kwargs):
        self.logger.debug(f"{self.__class__} ref={ref}, kwargs = {kwargs}\n")
        super().__init__(ref, **kwargs)

        self._default_config=dict(
            pipeline_args=dict(
                # return_text=None,
                # return_tensors=None,
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
                add_special_tokens=True,
            ),
            training_tokenizer_config=dict(
                padding_side='right',
                padding='longest',
                add_special_tokens=True,
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
    
    def get_model(self):

        model=getattr(self, 'model', None)
        if model is None:
            self.load_model()
            model=self.model
            del self.model
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

    def load_pipeline(self,**kwargs):
        self.pipeline = TextGenerationPipeline(model=self.model, tokenizer=self.inference_tokenizer,**kwargs)

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
        if not hasattr(self, 'inference_tokenizer'): self.load_inference_tokenizer()
        if not hasattr(self, 'pipeline'): self.load_pipeline()
    
    def get_pipeline_stream(self, the_input, **kwargs):

        streamer = TextIteratorStreamer(self.inference_tokenizer,skip_prompt=True)
        self.load_pipeline(streamer=streamer, **kwargs)

        generation_config=self.get_updated_config(kwargs, config_key='generation_config')
        self._recent_generation_config=generation_config

        pipeline_kwargs={
            'text_inputs':the_input,
            'tokenizer':self.pipeline.tokenizer,
            'generation_config':GenerationConfig.from_dict(generation_config),
            **self.get_updated_config(kwargs, config_key='pipeline_args')
        }
        
        streamer_thread=Thread(target=self.pipeline, kwargs=pipeline_kwargs)

        try:
            streamer_thread.start()
            for new_token in streamer:
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

    def get_pipeline_output(self, the_input, **kwargs):

        generation_config=self.get_updated_config(kwargs, config_key='generation_config')
        self._recent_generation_config=generation_config

        pipeline_kwargs={
            'text_inputs':the_input,
            'tokenizer':self.pipeline.tokenizer,
            'generation_config':GenerationConfig.from_dict(generation_config),
            **self.get_updated_config(kwargs, config_key='pipeline_args')
        }

        return self.pipeline(**pipeline_kwargs)
    
    def get_pred_df(self, df, **kwargs):

        if 'prompt_messages' in df.columns:
            the_input=list(df['prompt_messages'].values)
        
        elif 'prompt' in df.columns:
            the_input=list(df['prompt'].values)

        the_output=self.get_pipeline_output(the_input, **kwargs)

        df['pred_completion']=[o[0]['generated_text'] for o in the_output]
        return df

    def __call__(self, the_input, stream=False, use_chatformat=False, **kwargs):

        self.prepare_inference()

        if isinstance(the_input, pd.DataFrame):
            return self.get_pred_df(the_input, **kwargs)

        if use_chatformat:
            if isinstance(the_input, str):
                the_input=[{'role':'user','content':the_input}]
            
            elif isinstance(the_input, list) and all(isinstance(x, str) for x in the_input):
                the_input=[[{'role':'user','content':x}] for x in the_input]
   

        if stream:
            pred_completion=""
            for new_token in self.get_pipeline_stream(the_input, **kwargs):
                pred_completion+=new_token
                print(new_token,end='',flush=True)
            return pred_completion
        
        else:
            return self.get_pipeline_output(the_input, **kwargs)
        

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
                pad_token_id=128000,
            )
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

class MistralPipeline(LanguageModelPipeline):

    @staticmethod
    def matches(ref):
        if getattr_or_key(ref, 'name_or_path') == 'mistralai/Mistral-7B-Instruct-v0.2': return True
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
        
        self.update_config(self._default_config, overwrite_if_conflict=False)
        self.update_config_smart(
            kwargs, 
            interpret_none_as_val=True, 
            overwrite_if_conflict=True, 
            allow_new_atomic_keys=False, 
            allow_new_nested_keys=False
        )





class EmbeddingModelPipeline(BaseModelPipeline):
    type='emb_pipeline'

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