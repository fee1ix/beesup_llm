import pandas as pd
import warnings
from ..toolkit.setup_utils import *

def prepare_sample(sample, tokenizer):

    if isinstance(sample.get('prompt_messages'),list) and isinstance(sample.get('gold_message'),list):
        all_messages=sample['prompt_messages']+sample['gold_message']

        prompt_ids=tokenizer.apply_chat_template(sample['prompt_messages'],tokenize=True)
        prompt_len=len(prompt_ids)

        input_ids=tokenizer.apply_chat_template(sample['prompt_messages']+sample['gold_message'],tokenize=True)
        
        input_text=tokenizer.apply_chat_template(all_messages,tokenize=False)
        inputs=tokenizer.apply_chat_template(all_messages,return_dict=True)

        inputs['labels']=prompt_len*[-100]+input_ids[prompt_len:]
        
    elif pd.notna(sample.get('prompt')) and pd.notna(sample.get('gold_completion')):

        # if (not sample['prompt'].endswith('\n')) and (not sample['gold_completion'].startswith('\n')):
        #     warnings.warn('No newline between prompt and gold_completion --> Newline added!')
        #     input_text=sample['prompt']+'\n'+sample['gold_completion']
        
        input_text=sample['prompt']+sample['gold_completion']

        prompt_ids=tokenizer.encode(sample['prompt'],add_special_tokens=True)
        prompt_len=len(prompt_ids)

        input_ids=tokenizer.encode(sample['prompt']+sample['gold_completion'],add_special_tokens=True)#+[tokenizer.eos_token_id]

        inputs=tokenizer(input_text)
        inputs['labels']=prompt_len*[-100]+input_ids[prompt_len:]

        input_text=tokenizer.decode(inputs['input_ids'])
    
    elif isinstance(sample.get('prompt_messages'),list) :
        input_text=tokenizer.apply_chat_template(sample['prompt_messages'],tokenize=False)
        inputs=tokenizer.apply_chat_template(sample['prompt_messages'],return_dict=True)

    elif pd.notna(sample.get('prompt')):
        
        inputs=tokenizer(sample['prompt'])
        input_text=tokenizer.decode(inputs['input_ids'])
    
    else:
        raise Warning("undefined sample format")

    return {**inputs}

def prepare_sample_for_chat_completion(the_input, tokenizer):

    if isinstance(the_input, list): #assume that chat messages are given
        prompt_messages=the_input
    
    elif hasattrs_or_keys(the_input, ['prompt_messages']):
        prompt_messages=the_input['prompt_messages']

    return tokenizer.apply_chat_template(prompt_messages,return_dict=True)

def prepare_sample_for_chat_finetuning(the_input, tokenizer):
    
    if hasattrs_or_keys(the_input, ['prompt_messages','gold_message']):
        prompt_messages=getattr_or_key(the_input,'prompt_messages')
        gold_message=getattr_or_key(the_input,'gold_message')

    
    if gold_message[0]['role']!='assistant': warnings.warn('The gold message should be an assistant message')
    if prompt_messages[-1]['role']!='user': warnings.warn('The last prompt message should be a user message')

    all_messages=prompt_messages+gold_message

    prompt_ids=tokenizer.apply_chat_template(prompt_messages,tokenize=True)
    prompt_len=len(prompt_ids)

    input_ids=tokenizer.apply_chat_template(all_messages,tokenize=True)
    
    #input_text=tokenizer.apply_chat_template(all_messages,tokenize=False)
    inputs=tokenizer.apply_chat_template(all_messages,return_dict=True)

    inputs['labels']=prompt_len*[-100]+input_ids[prompt_len:]

    return inputs





def to_outputs_df(generation_outputs, tokenizer=None):
    if tokenizer is not None:
        min_id, max_id=min(tokenizer.vocab.values()), max(tokenizer.vocab.values())
    else:
        min_id,max_id=0, 128255

    data=[]
    for i in range(len(generation_outputs['all_input_ids'])):
        row=dict()

        for col in ['all_input_ids', 'all_label_ids', 'all_all_ids', 'all_losses']:
            if isinstance(generation_outputs[col],type(None)): continue

            values=generation_outputs[col][i]
            values=values[(values >= min_id) & (values <= max_id)]
            row[col[4:]]=values

        row['pred_ids']=row['all_ids'][len(row['input_ids']):]

        if tokenizer is not None:
            row['pred_completion']=tokenizer.decode(row['pred_ids'], skip_special_tokens=True)

        data.append(row)

    outputs_df=pd.DataFrame(data)
    
    return outputs_df