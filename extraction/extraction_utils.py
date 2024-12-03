import re
import json
import copy
import pandas as pd
from pydantic import ValidationError

from beesup_llm.extraction import \
    ExtractionScheme4SingleObservation, \
    ExtractionScheme4MultipeObservations, \
    OBSERVATION_ATTRIBUTES_DICT, \
    OBSERVATION_DESCRIPTIONS_DICT, \
    FEW_SHOT_EXAMPLES, \
    EXTRACTION_PROMPT

def get_gold_completion(gold_json):
    return f"```json\n{json.dumps(gold_json,indent=2,ensure_ascii=False)}\n```"

def get_prompt_messages(report_passage, use_extraction_prompt=True, use_few_shots=True, **kwargs):

    prompt_messages=[]

    if use_extraction_prompt:

        prompt_messages.extend([
            {
            'role':'user',
            'content':EXTRACTION_PROMPT
            },
            {
            'role':'assistant',
            'content':'Ok give me a report passage!' #need this message to assure alternation between user and assistant messages
            }
        ])
    
    if use_few_shots:

        for example in FEW_SHOT_EXAMPLES:
            prompt_messages.append({
                'role':'user',
                'content':'REPORT-PASSAGE:\n{report_passage}'.format(**example)
            })

            prompt_messages.append({
                'role':'assistant',
                'content':get_gold_completion(example['gold_json'])
            })

    prompt_messages.append({
        'role':'user',
        'content':f'REPORT-PASSAGE:\n{report_passage}'
    })

    return prompt_messages

def pydantic_parse(completion,exclude_none=True):

    is_valid=False
    is_empty=False

    if completion==None: completion="```json\n{}\n```"

    if not isinstance(completion,str): return None, is_valid, is_empty

    completion=completion.strip()
    re_match=re.search(r"```json\s*(.*?)```",completion,flags=re.S|re.M)


    if not re_match: return None, is_valid, is_empty

    completion=re_match.groups()[0]

    try:
        completion_json=ExtractionScheme4MultipeObservations.parse_raw(completion).dict(exclude_none=exclude_none)

        scientific_name_in_meta = ('meta_scientific_name' in completion_json)

        scientific_name_in_obs = False
        if 'observations' in completion_json:
            # if not isinstance(completion_json['observations'],list):
            #     completion_json['observations']

            if isinstance(completion_json['observations'],list):
                scientific_name_in_obs = all(['scientific_name' in obs.keys() for obs in completion_json['observations']])

        is_valid = scientific_name_in_meta or scientific_name_in_obs
        is_empty = all([val==None for val in completion_json.values()])

        return completion_json, is_valid, is_empty
    
    except ValidationError as e:
        return str(e), is_valid, is_empty

def integrate_metas(base_json,allow_extra_keys=False):

    meta_row={k.replace('meta_',''):v for k,v in base_json.items() if k != 'observations'}
    obs_rows=base_json['observations']

    if obs_rows==None: return [meta_row]

    for i,obs_row in enumerate(obs_rows):

        for key in meta_row.keys():

            meta_val=meta_row.get(key)
            obs_val=obs_row.get(key)

            if pd.notna(meta_val) & pd.isna(obs_val):
                obs_rows[i][key]=meta_val

            elif pd.notna(meta_val) & pd.notna(obs_val):
                obs_rows[i][key]=meta_val+", "+obs_val
            

    for i,obs_row in enumerate(obs_rows):
        obs_rows[i]={k:obs_row[k] for k in OBSERVATION_ATTRIBUTES_DICT.keys() if k in obs_row}

        if allow_extra_keys:
            for key in [k for k in obs_row.keys() if k not in OBSERVATION_ATTRIBUTES_DICT.keys()]:
                obs_rows[i][key]=obs_row[key]

    return obs_rows

def recursive_remove_none(dictionary):
    keys_to_remove = [key for key, val in dictionary.items() if val is None]
    for key in keys_to_remove:
        del dictionary[key]
    
    for key, val in list(dictionary.items()):  # Use list to avoid dictionary changed size during iteration error
        if isinstance(val, dict):
            recursive_remove_none(val)

def isna(v):
    if isinstance(v,list):
        return all(pd.isna(v))
    else:
        return pd.isna(v)

def stringify_json(old_json):
    new_json=dict()
    for k,v in old_json.items():
        if not isna(v):
            new_json[k]=str(v)
        else:
            new_json[k]=None
    return new_json

def stringify_extraction_json(old_json):
    
    new_json=dict()
    if "observations" in old_json:
        observations=[stringify_json(o) for o in old_json['observations']]
    else:
        observations=None

    new_json=stringify_json(old_json)
    new_json['observations']=observations

    return new_json

def normalize_json(old_json,ordered_keys):
    new_json=dict()

    for key in ordered_keys:
        if key in old_json:
            new_json[key]=old_json[key]
        else:
            new_json[key]=None
    return new_json

def normalize_extraction_json(old_json):

    new_json=dict()

    #if "observations" in old_json:
    if isinstance(old_json.get('observations',None),list):
        observations=[normalize_json(o,ExtractionScheme4SingleObservation.model_fields.keys()) for o in old_json['observations']]
    else:
        observations=None

    new_json=normalize_json(old_json,ExtractionScheme4MultipeObservations.model_fields.keys())
    new_json['observations']=observations

    return new_json

def denormalize_json(old_json):
    return {k:v for k,v in old_json.items() if v!=None}

def denormalize_extraction_json(old_json):
    if isinstance(old_json['observations'],list):
        old_json['observations']=[denormalize_json(o) for o in old_json['observations']]
    
    old_json=denormalize_json(old_json)
    return old_json



def tabelize_json(base_json, create_meta_row=True):

    if not isinstance(base_json,dict):
        return pd.DataFrame(columns=ExtractionScheme4SingleObservation.model_fields.keys())
    
    base_json=normalize_extraction_json(base_json)
    base_json=copy.deepcopy(base_json)
    meta_row={k.replace('meta_',''):v for k,v in base_json.items() if k != "observations"}
    meta_emty=all([val==None for val in meta_row.values()])

    #print(base_json)

    obs_rows=base_json["observations"]
    obs_emty=(obs_rows==None)

    if meta_emty and obs_emty: #all emty
        table_df=pd.DataFrame(columns=meta_row.keys()) #emty dataframe only with headers
    
    elif not meta_emty and not obs_emty:

        if create_meta_row:
            table_df=pd.DataFrame([meta_row]+obs_rows)
        else:
            table_df=pd.DataFrame(integrate_metas(base_json))
    

    elif obs_emty: #only obs_rows emty

        if create_meta_row:
            table_df=pd.DataFrame([meta_row])
        else:
            table_df=pd.DataFrame(integrate_metas(base_json))

    elif meta_emty: #only meta_row emty

        if create_meta_row:
            table_df=pd.DataFrame([meta_row]+obs_rows)
        else:
            table_df=pd.DataFrame(integrate_metas(base_json))
    else:
        raise ValueError('undefined case!')
    
    table_df.attrs['has_meta_row']=create_meta_row

    return table_df


def parse_completion(completion,verbose=False):
    base_json=pydantic_parse(completion,exclude_none=False)
    #base_json=stringify_extraction_json(base_json)

    if pydantic_parse.is_valid==False:
        base_json={k:None for k in ExtractionScheme4MultipeObservations.model_fields.keys()}

    raw_df=tabelize_json(base_json,create_meta_row=True)
    raw_df.attrs['is_valid']=pydantic_parse.is_valid
    raw_df.attrs['is_empty']=pydantic_parse.is_empty

    tab_df=tabelize_json(base_json,create_meta_row=False)
    tab_df.attrs['is_valid']=pydantic_parse.is_valid
    tab_df.attrs['is_empty']=pydantic_parse.is_empty
    tab_df.index=tab_df.index+1


    if verbose:
        print(f'completion is parsable: {pydantic_parse.is_valid}')
        print(f'   completion is empty: {pydantic_parse.is_empty}')

    parse_completion.pred_json=base_json
    parse_completion.is_valid=pydantic_parse.is_valid
    parse_completion.is_empty=pydantic_parse.is_empty

    
    return raw_df,tab_df

