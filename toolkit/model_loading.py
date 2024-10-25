import logging
logger=logging.getLogger(__name__)

def load_base_model_and_tokenizer(config,verbose=False):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    logger.info(f'START {config['base_model']}')


    bnb_config = BitsAndBytesConfig(
        bnb_4bit_compute_dtype=torch.bfloat16,
        **config['bnb_config']
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        config['base_model'],
        device_map="auto",
        quantization_config=bnb_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config['base_model'],
        **config['tokenizer_config']
    )

    logger.info(f'END {config['base_model']}')

    return model, tokenizer

def load_peft_model_and_tokenizer(config,verbose=False):

    logger.info(f'START {config['base_model']} + {config['peft_model']}')

    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer, BitsAndBytesConfig
    
    bnb_config = BitsAndBytesConfig(
        bnb_4bit_compute_dtype=torch.bfloat16,
        **config['bnb_config']
        )
    
    model = AutoPeftModelForCausalLM.from_pretrained(
        f'{config["training_config"]["path"]}/{config["peft_model"]}',
        device_map='auto',
        quantization_config=bnb_config
    )

    tokenizer = AutoTokenizer.from_pretrained(
        f'{config["training_config"]["path"]}/{config["peft_model"]}',
        **config['tokenizer_config']
        )

    logger.info(f'END {config['base_model']} + {config['peft_model']}')
    
    return model,tokenizer
