import torch
import logging
import pandas as pd

from typing import Union

from beesup_llm import get_labhandler, _isinstance

from threading import Thread
from transformers import \
    AutoTokenizer, \
    AutoModelForCausalLM, \
    BitsAndBytesConfig, \
    GenerationConfig, \
    TextGenerationPipeline, \
    TextIteratorStreamer

logging.getLogger("transformers").setLevel(logging.ERROR)




class LLMPipeline(object):

    logger = logging.getLogger(__name__)

    @classmethod
    def from_model(cls, model=None):

        name_or_path = getattr(model, 'name_or_path', None)
        if name_or_path in ['meta-llama/Meta-Llama-3.1-8B-Instruct']:
            return LlamaPipeline(model=model, name_or_path=name_or_path)

        return cls(model=model, name_or_path=name_or_path)

    def __init__(self, ref=None, labh=get_labhandler(), **kwargs):

        if 'model' in kwargs:
            self.model=kwargs.pop('model')
            self.name_or_path=getattr(self.model, 'name_or_path', None)

        self.name_or_path = getattr(self, 'name_or_path', None) or  kwargs.get('name_or_path', None)

        self.generation_config=kwargs.get('generation_config',dict())
        self.generation_config['return_dict_in_generate']=False
        self.generation_config['max_time']=kwargs.get('max_time',600)
        self.generation_config['do_sample']=kwargs.get('do_sample',False)
        self.generation_config['stop_strings']=kwargs.get('stop_strings',None)
        self.generation_config['max_new_tokens']=kwargs.get('max_new_tokens',1000)

        self.inference_tokenizer_config=kwargs.get('inference_tokenizer_config',dict())
        self.inference_tokenizer_config['padding']='longest'
        self.inference_tokenizer_config['padding_side']='left'

        self.training_tokenizer_config=dict()
        self.training_tokenizer_config['padding']='longest'
        self.training_tokenizer_config['padding_side']='right'

        self.pipeline_args=dict()
        self.pipeline_args['return_full_text']=False
        self.pipeline_args['clean_up_tokenization_spaces']=True

        if labh is not None:
            self.labh=labh
            self.labh.attach_parent(locals())

    def load_model(self):
        self.logger.info(f"Loading model {self.name_or_path}")
        self.model=AutoModelForCausalLM.from_pretrained(
            self.name_or_path,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(
                bnb_4bit_compute_dtype=torch.bfloat16,
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
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
    
    def count_tokens(self, pipe_input):
        tokenizer=self.get_tokenizer()
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


        pipeline_kwargs={
            'text_inputs':pipe_input,
            'tokenizer':self.pipeline.tokenizer,
            'generation_config':GenerationConfig.from_dict(self.generation_config),
            **self.pipeline_args
        }
        
        streamer_thread=Thread(target=self.pipeline, kwargs=pipeline_kwargs)
        self._yielded_tokens=[]
        try:
            streamer_thread.start()
            
            for new_token in streamer:
                self._yielded_tokens.append(new_token)
                new_token=new_token.replace(self.inference_tokenizer.eos_token,'') # remove eos_token

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

        self.prepare_inference()

        pipeline_kwargs={
            'text_inputs':pipe_input,
            'tokenizer':self.pipeline.tokenizer,
            'generation_config':GenerationConfig.from_dict(self.generation_config),
            **self.pipeline_args
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

class LlamaPipeline(LLMPipeline):

    def __init__(self, ref=None, **kwargs):
        super().__init__(ref, **kwargs)

        self.name_or_path = getattr(self,'name_or_path',None) or 'meta-llama/Meta-Llama-3.1-8B-Instruct'

        self.generation_config['pad_token']='<|begin_of_text|>'
        self.generation_config['pad_token_id']=128000

        #self.inference_tokenizer_config=dict()
        self.inference_tokenizer_config['max_length']=8192
        self.inference_tokenizer_config['pad_token']='<|begin_of_text|>'
        self.inference_tokenizer_config['pad_token_id']=128000

        #self.training_tokenizer_config=dict()
        self.training_tokenizer_config['max_length']=8192
        self.training_tokenizer_config['pad_token']='<|end_of_text|>'
        self.training_tokenizer_config['pad_token_id']=128001
    
