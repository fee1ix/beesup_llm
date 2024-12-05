from beesup_llm.model import *


from transformers import AutoModel

class EmbModelWrap(BaseModelWrap):
    type='emb_model'

    @staticmethod
    def matches(ref):
        if getattr_or_key(ref, 'type') == 'emb_model': return True
        if JinaaiModelWrap.matches(ref): return True
        return False
    
    @classmethod
    def from_ref(cls, ref=None, **kwargs):
        kwargs.update(get_cls_attrs(cls))
        cls.logger.debug(f"{cls} ref={ref}, kwargs = {kwargs}\n")

        pre_config = get_config_from_ref(ref, **kwargs)

        if JinaaiModelWrap.matches(pre_config):
            cls.logger.debug(f"It's a JinaaiModelWrap!")
            return JinaaiModelWrap(ref=pre_config, **kwargs)

        return cls(ref=pre_config, **kwargs)


    def __init__(self, ref=None, **kwargs):
        self.logger.debug(f"{self.__class__} ref={ref}, kwargs = {kwargs}\n")
        super().__init__(ref,**kwargs)

        self._config_key_order.extend([])
        self._config_keys_to_exclude.extend([])

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

        self.update_config(self._default_config, overwrite_if_conflict=False)
    
    def load_model(self,**kwargs):

        self.logger.info(f"Loading model {self.name_or_path}")

        bnb_config=self.get_updated_config(kwargs, config_key='bnb_config')
        model_load_config=self.get_updated_config(kwargs, config_key='model_load_config')

        if bnb_config:
            quantization_config=BitsAndBytesConfig(
                bnb_4bit_compute_dtype=torch.bfloat16,
                **bnb_config
                )
        else:
            quantization_config=None


        self.model=AutoModel.from_pretrained(
            self.name_or_path,
            device_map="auto",
            quantization_config=quantization_config,

            **model_load_config,
        )

        # self.model=AutoModel.from_pretrained(
        #     self.name_or_path,
        #     trust_remote_code=self.trust_remote_code,
        #     use_flash_attn=self.use_flash_attn,
        #     ).to('cuda')

        return

    # Add batching to your encode process
    def batch_encode(self, chunks, batch_size=128, **kwargs):
        embs = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            embs.append(self.model.encode(batch, **kwargs))
            torch.cuda.empty_cache()

        embs=torch.cat(embs, dim=0)
        embs=embs.to('cpu')
        return list(embs.numpy())
        
    def unique_encode(self, chunks, **kwargs):

        # Step 1: Remove redundancy by creating a set of unique chunks
        unique_chunks = list(set(chunks))

        # Step 2: Encode each unique chunk
        unique_embs=self.batch_encode(unique_chunks, **kwargs)
        #unique_embs = self.encode(unique_chunks, **kwargs)
        
        # Step 3: Create a mapping from each unique chunk to its embedding
        emb_dict = dict(zip(unique_chunks, unique_embs))
        
        # Step 4: Map the embeddings back to the original chunk list
        embs = [emb_dict[chunk] for chunk in chunks]

        return embs


    def encode(self, chunks, **kwargs):

        encode_config=self.get_updated_config(kwargs, config_key='encode_config')

        embs=self.model.encode(
            chunks,
            **encode_config
        )

        if hasattr(embs,'device'):
            if embs.device.type=='cuda':
                embs=embs.to('cpu')


        return embs

class JinaaiModelWrap(EmbModelWrap):

    @staticmethod
    def matches(ref):
        if getattr_or_key(ref, 'name_or_path') == 'jinaai/jina-embeddings-v3': return True
        return False

    def __init__(self, ref=None, **kwargs):
        self.logger.debug(f"{self.__class__} ref={ref}, kwargs = {kwargs}\n")
        super().__init__(ref, **kwargs)

        self._config_key_order.extend([])
        self._config_keys_to_exclude.extend([])

        self._default_config=dict(
            name_or_path='jinaai/jina-embeddings-v3',
            bnb_config=None,
            model_load_config=dict(
                use_flash_attn=False,
                task='separation',
            ),
            encode_config=dict(
            )
        )
        self.update_config(self._default_config, overwrite_if_conflict=False)

class NvidiaModelWrap(EmbModelWrap):

    @staticmethod
    def matches(ref):
        if getattr_or_key(ref, 'name_or_path') == 'nvidia/NV-Embed-v2': return True
        return False
    

    def __init__(self, ref=None, **kwargs):
        self.logger.debug(f"{self.__class__} ref={ref}, kwargs = {kwargs}\n")
        super().__init__(ref, **kwargs)

        self._config_key_order.extend([])
        self._config_keys_to_exclude.extend([])

        self._default_config=dict(
            name_or_path='nvidia/NV-Embed-v2',
        
            model_load_config=dict(
            ),

            encode_config=dict(
                max_length = 32768,
                instruction="Instruct: Retrieve the a suitable header for the chunk.\nChunk: ",
            ),
        )
        self.update_config(self._default_config, overwrite_if_conflict=False)

    # def load_model(self):

    #     self.logger.info(f"Loading model {self.name_or_path}")
    #     self.model=AutoModelForCausalLM.from_pretrained(
    #         self.name_or_path,
    #         device_map="auto",
    #         quantization_config=BitsAndBytesConfig(
    #             bnb_4bit_compute_dtype=torch.bfloat16,
    #             **self.bnb_config
    #             ),
    #     )

    #     return
    
    # def encode(self, chunks, **kwargs):

    #     encode_kwargs = dict(
    #         instruction=self.instruction,
    #     )
    #     encode_kwargs.update(kwargs)

    #     embs=self.model.encode(
    #         chunks,
    #         **encode_kwargs
    #     )

    #     return embs

