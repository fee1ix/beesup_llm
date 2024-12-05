
from beesup_llm.model import *

from threading import Thread
from transformers import TextGenerationPipeline, TextIteratorStreamer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig, set_seed


class GenerationModelWrap(BaseModelWrap):

    type='gen_model'
    
    @staticmethod
    def matches(ref):
        if getattr_or_key(ref, 'type') == 'gen_model': return True

        if PeftLlamaModelWrap.matches(ref): return True
        if LlamaModelWrap.matches(ref): return True
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

        # if PeftLlamaModelWrap.matches(pre_config):
        #     return PeftLlamaModelWrap.from_ref(ref=pre_config, **kwargs)

        if LlamaModelWrap.matches(pre_config):
            return LlamaModelWrap(ref=pre_config, **kwargs)
        
        return cls(ref=pre_config, **kwargs)
    
    def __init__(self, ref=None, **kwargs):
        self.logger.debug(f"{self.__class__} ref={ref}, kwargs = {kwargs}\n")
        super().__init__(ref, **kwargs)

        self._config_key_order.extend([])
        self._config_keys_to_exclude.extend(['inference_tokenizer','training_tokenizer','outputs'])

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
                return_dict_in_generate=False,
                max_new_tokens=750,
                max_time=600,
                stop_strings=None,
            ),
            data_collator_config=dict(
                label_pad_token_id=-100,
            ),
            dataloader_config=dict(
                batch_size=4,
            )
        )
        
        self.update_config(self._default_config, overwrite_if_conflict=False)

    def prepare_training(self):
        
        if not hasattr(self, 'model'): self.load_model()
        if not hasattr(self, 'training_tokenizer'): self.load_training_tokenizer()

    def prepare_inference(self):

        if not hasattr(self, 'model'): self.load_model()
        if not hasattr(self, 'inference_tokenizer'): self.load_inference_tokenizer()

    def prepare_inference_inputs(self, the_input):

        from transformers.tokenization_utils_base import BatchEncoding

        inputs=None
        inference_tokenizer=self.inference_tokenizer

        if isinstance(the_input, str): 
            inputs=inference_tokenizer(the_input,return_tensors='pt').to('cuda')
            
        if isinstance(the_input, list):
            if all(isinstance(item, str) for item in the_input):
                inputs=inference_tokenizer.batch_encode_plus(the_input,padding=True, return_tensors='pt').to('cuda')
            
            elif all(isinstance(item, dict) for item in the_input):
                inputs=inference_tokenizer.apply_chat_template(the_input, return_dict=True,return_tensors='pt').to('cuda')
            
            elif all(isinstance(item,int) for item in the_input):
                inputs=inference_tokenizer.pad({"input_ids": [the_input]}, padding=True,return_tensors='pt').to('cuda')
            
            elif all(isinstance(item,list) for item in the_input):

                if all(all(isinstance(item, int) for item in item_list) for item_list in the_input):
                    inputs=inference_tokenizer.pad({"input_ids": the_input}, padding=True,return_tensors='pt').to('cuda')
                    
        if isinstance(the_input, dict):
            if 'input_ids' in the_input.keys():
                inputs={k:v for k,v in the_input.items() if k in ['input_ids','attention_mask']}
        
        return BatchEncoding(inputs)
    
    def get_generation_config(self, **kwargs):

        generation_config=self.generation_config
        if kwargs.get('generation_config'):
            generation_config.update(**kwargs.get('generation_config'))
        else:
            kwargs=filter_kwargs(kwargs, ref=self.generation_config)
            generation_config.update(**kwargs)

        generation_config=GenerationConfig.from_dict(generation_config)

        #self.logger.info(f'generation_config: {generation_config}')

        return generation_config


    def prepare_streaming(self):
        if not hasattr(self, 'streamer'):
            self.streamer = TextIteratorStreamer(self.inference_tokenizer,skip_prompt=True)

        if not hasattr(self, 'pipeline'):
            self.pipeline = TextGenerationPipeline(model=self.model, tokenizer=self.inference_tokenizer, streamer=self.streamer)

    def generation_step_pipeline(self, input_text_or_messages, **kwargs):

        self.pipeline = TextGenerationPipeline(model=self.model, tokenizer=self.inference_tokenizer)

        generation_config=self.get_updated_config(kwargs, config_key='generation_config')
        self._recent_generation_config=generation_config

        pipeline_kwargs={
            'text_inputs':input_text_or_messages,
            'tokenizer':self.pipeline.tokenizer,
            'generation_config':GenerationConfig.from_dict(generation_config),
        }
        result=self.pipeline(**pipeline_kwargs)

        return result[0]['generated_text'][-1]['content']

    def generation_stream(self, input_text_or_messages, stop_event=None, **kwargs):

        self.prepare_inference()
        self.prepare_streaming()

        generation_config=self.get_updated_config(kwargs, config_key='generation_config')
        self._recent_generation_config=generation_config


        pipeline_kwargs={
            'text_inputs':input_text_or_messages,
            'tokenizer':self.pipeline.tokenizer,
            'generation_config':GenerationConfig.from_dict(generation_config),
        }
        
        streamer_thread=Thread(target=self.pipeline, kwargs=pipeline_kwargs)

        try:
            streamer_thread.start()
            for new_token in self.streamer:
                yield new_token

            streamer_thread.join()
        
        except Exception as e:
            self.logger.info(f'{e}')
            streamer_thread.join()
            torch.cuda.empty_cache()
            self.logger.info(f'generation_stream: executed torch.cuda.empty_cache()')
        
        del self.streamer
        del self.pipeline
        
        return

    def get_data_collator(self, **kwargs):

        from transformers import DataCollatorForSeq2Seq
        data_collator_config=self.data_collator_config

        # if kwargs.get('data_collator_config'): 
        #     data_collator_config.update(kwargs.get('data_collator_config'))
        # else:
        #     kwargs=filter_kwargs(kwargs, ref=data_collator_config)
        #     data_collator_config.update(kwargs)

        data_collator_config=self.get_updated_config(kwargs, config_key='data_collator_config')
        
        data_collator=DataCollatorForSeq2Seq(
            model=self.model,
            tokenizer=self.get_inference_tokenizer(),
            **data_collator_config
            )

        return data_collator

    def get_dataloader(self, ds, **kwargs):

        data_collator=kwargs.get('data_collator', self.get_data_collator(**kwargs))
        
        from torch.utils.data import DataLoader
        dataloader_config=self.get_updated_config(kwargs, config_key='dataloader_config')

        dataloader=DataLoader(
            ds,
            collate_fn=data_collator,
            **dataloader_config
            )

        return dataloader



    def generation_step(self, inputs, **kwargs):

        generation_config=self.get_updated_config(kwargs, config_key='generation_config')
        self._recent_generation_config=generation_config


        #self.logger.info(f"generation_config: {generation_config}")

        # stop_sequence=generation_config.get('stop_sequence',None)
        # if stop_sequence is not None:
        #     stop_sequence_ids = self.inference_tokenizer.encode(stop_sequence, add_special_tokens=False)
        #     if len(stop_sequence_ids) > 1:
        #         warnings.warn(
        #             "Stopping on a multiple token sequence is not yet supported on transformers. The first token of"
        #             " the stop sequence will be used as the stop sequence string in the interim."
        #         )
            
        #     del generation_config['stop_sequence']
        #     inputs['eos_token_id']=stop_sequence_ids[0]
        

        inputs.to('cuda')

        outputs=self.model.generate(
            generation_config=GenerationConfig.from_dict(generation_config),
            tokenizer=self.inference_tokenizer,
            **inputs)
        
        self._outputs=outputs

        return outputs


    def generation_loop(self, dataset, **kwargs):

        self.prepare_inference()

        self.logger.info(f"kwargs: {kwargs}")

        dataloader=kwargs.get('dataloader', self.get_dataloader(dataset,**kwargs))
        self._dataloader=dataloader

        from transformers.trainer_pt_utils import EvalLoopContainer

        # Initialize containers
        all_input_ids = EvalLoopContainer(do_nested_concat=True, padding_index=-100)
        all_label_ids = EvalLoopContainer(do_nested_concat=True, padding_index=-100)
        all_pred_ids = EvalLoopContainer(do_nested_concat=True, padding_index=-100)
        all_all_ids = EvalLoopContainer(do_nested_concat=True, padding_index=-100)
        all_losses = EvalLoopContainer(do_nested_concat=True, padding_index=-100)

        num_batches=len(dataloader)

        for step, inputs in enumerate(dataloader):

            input_ids, label_ids, pred_ids,all_ids, losses  = None, None, None, None, None
            
            input_ids=inputs.get('input_ids',None)
            label_ids=inputs.get('labels',None)

            self.logger.info(f'batch {step+1}/{num_batches}')
            all_ids=self.generation_step(inputs, **kwargs)

     
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

    def generate(self, inputs, **kwargs):
        if kwargs.get('seed'): set_seed(kwargs.get('seed'))

        generation_config=self.get_generation_config(**kwargs)

        outputs=self.model.generate(
            generation_config=generation_config,
            tokenizer=self.inference_tokenizer,
            **inputs)

        return outputs

    def get_pred_completions(self, inputs, outputs):

        pred_completions=[]
        for input_ids, output_ids in zip(inputs['input_ids'], outputs):
            pred_ids=output_ids[len(input_ids):]
            pred_completion=self.inference_tokenizer.decode(pred_ids, skip_special_tokens=True)
            pred_completions.append(pred_completion)
        
        return pred_completions

    def get_inputs_from_str(self, the_input, as_chat=False):

        tokenizer=self.get_inference_tokenizer()

        if as_chat:
            inputs=tokenizer.apply_chat_template([{'role':'user','content':the_input}], return_dict=True,return_tensors='pt').to('cuda')
        else:
            inputs=tokenizer(the_input,return_tensors='pt').to('cuda')
        
        return inputs
    
    def get_pred(self, the_input, stream=False, **kwargs):

        if stream:
            pred_completion=""
            for new_token in self.generation_stream(input_text_or_messages=inputs, **kwargs):
                pred_completion += new_token
                print(new_token,end='',flush=True)
        else:
            inputs=self.get_inputs_from_str(the_input)
            outputs=self.generation_step(inputs, **kwargs)
            pred_completion=self.get_pred_completions(inputs, outputs)[0]

            return pred_completion
    
    def get_pred_chat(self, messages, stream=False, **kwargs):

        if stream:
            pred_completion=""
            for new_token in self.generation_stream(input_text_or_messages=messages, **kwargs):
                pred_completion += new_token
                print(new_token,end='',flush=True)
        else:
            tokenizer=self.get_inference_tokenizer()
            inputs=tokenizer.apply_chat_template(messages, return_dict=True,return_tensors='pt').to('cuda')
            outputs=self.generation_step(inputs, **kwargs)
            pred_completion=self.get_pred_completions(inputs, outputs)[0]
        
        return pred_completion

    def get_pred_df(self, df, **kwargs):
        from datasets import Dataset
        tokenizer=self.get_inference_tokenizer()

        if 'prompt_messages' in df.columns:
            ds=Dataset.from_list(df.apply(lambda x: prepare_sample_for_chat_completion(x, tokenizer),axis=1).to_list())
            outputs=self.generation_loop(ds,**kwargs)
            outputs_df=to_outputs_df(outputs,tokenizer=tokenizer)
            df['pred_completion']=outputs_df['pred_completion'].values

        return df


    def __call__(self, the_input, stream=False, as_chat=False, **kwargs):

        self.prepare_inference()

        if isinstance(the_input, str):

            if as_chat:
                chat_messages=[{'role':'user','content':the_input}]
                #return self.get_pred_chat(chat_messages, stream=stream, **kwargs)
                return self.generation_step_pipeline(input_text_or_messages=chat_messages, **kwargs)
            else:
                return self.get_pred(the_input, stream=stream, as_chat=as_chat, **kwargs)


        elif isinstance(the_input, list):
            return self.get_pred_chat(the_input, stream=stream, **kwargs)

        
        elif isinstance(the_input, pd.DataFrame):
            return self.get_pred_df(the_input, **kwargs)


        # self.prepare_inference()
        # inputs=self.prepare_inference_inputs(input)
        # is_batch=inputs['input_ids'].shape[0]>1

        # if stream and not is_batch:
        #     pred_completion=""
        #     for new_token in self.generate_stream(input_text_or_messages=inputs, **kwargs):
        #         pred_completion += new_token
        #         print(new_token,end='',flush=True)
        
        # else:
        #     outputs=self.generate(inputs, **kwargs)
        #     self.outputs=outputs # for debugging
        #     pred_completions=self.get_pred_completions(inputs, outputs)

        # if not is_batch:
        #     return pred_completions[0]
        # else:
        #     return pred_completions
        


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
            
class LlamaModelWrap(GenerationModelWrap):

    @staticmethod
    def matches(ref):
        if getattr_or_key(ref, 'name_or_path') == 'meta-llama/Meta-Llama-3.1-8B-Instruct': return True
        return False

    def __init__(self, ref=None, **kwargs):
        self.logger.debug(f"{self.__class__} ref={ref}, kwargs = {kwargs}\n")
        super().__init__(ref, **kwargs)

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
        
        self.update_config(self._default_config, overwrite_if_conflict=False)

from peft import AutoPeftModelForCausalLM
class PeftLlamaModelWrap(LlamaModelWrap):

    @staticmethod
    def matches(ref):
        raise Warning("PeftLlamaModelWrap is not implemented yet.")
        return False

    def __init__(self, ref=None, **kwargs):
        super().__init__(ref,**kwargs)

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
