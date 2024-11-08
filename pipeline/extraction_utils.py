import os
import re
import sys
import json
import math
import copy
import difflib
import pandas as pd
from rapidfuzz import fuzz
from functools import reduce
import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import matplotlib.lines as mlines
# import matplotlib.colors as mcolors
# from transformers import AutoTokenizer

from ..toolkit.visualization import * 






def print_sample(sample,headers=['gold_completion','pred_completion']):

    print(f"{sample.name} | {sample.get('test_logit_ref')} | style: {sample.get('report_style')}")

    html_content=""
    html_content+=f"""
<div style="float: left;">
    <h3>Report Passage (style: {sample.report_style})</h3>
    <pre style="white-space: pre-wrap;">{sample.report_passage}</pre>
</div>
"""
    display(HTML(html_content))
    
    texts=[sample[h] for h in headers]
    headers=[f"<h2>{h}</h2>" for h in headers]

    print_multicol(headers)
    print_multicol(texts)

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
    if "observations" in old_json:
        observations=[normalize_json(o,ObservationScheme.model_fields.keys()) for o in old_json['observations']]
    else:
        observations=None

    new_json=normalize_json(old_json,ReportScheme.model_fields.keys())
    new_json['observations']=observations

    return new_json

def denormalize_json(old_json):
    return {k:v for k,v in old_json.items() if v!=None}

def denormalize_extraction_json(old_json):
    if isinstance(old_json['observations'],list):
        old_json['observations']=[denormalize_json(o) for o in old_json['observations']]
    
    old_json=denormalize_json(old_json)
    return old_json

#EVALUATION UTILS


# MATCHING OBSERVATIONS PRED vs. GOLD
def fuzzy_match(val_gold,val_pred):

    #only consinder cols where both attributes are defined for matching!!

    #TP: fuzzy score if both values are defined
    if (pd.notna(val_gold)) and (pd.notna(val_pred)):
        val_gold=str(val_gold).strip()
        val_pred=str(val_pred).strip()
        #score=0.1+0.009*fuzz.ratio(val_gold,val_pred) #always > 0.1
        score=fuzz.ratio(val_gold,val_pred)/100

    # elif (pd.isna(val_gold)) and (pd.isna(val_pred)):
    #     #score=1.0
    #     score=torch.nan
    
    # #FP:
    # elif (pd.isna(val_gold)) and (pd.notna(val_pred)):
    #     score=0.0

    # #FN:
    # elif (pd.notna(val_gold)) and (pd.isna(val_pred)):

    else:
        #score=0.0
        score=torch.nan
    

    return score

def cross_match(gold_df,pred_df):

    # gold_keys=set(gold_df.dropna(axis=1, how='all').columns)
    # pred_keys=set(pred_df.dropna(axis=1, how='all').columns)
    # intersection_keys=list(gold_keys and pred_keys)

    match_data=[]
    for g,gold_row in gold_df.iterrows():
        for p,pred_row in pred_df.iterrows():
            match_row={
                'g':g,
                'p':p,
                'scientific_name_gold':gold_row['scientific_name'],
                'scientific_name_pred':pred_row['scientific_name']
                }
            
            for key in pred_df.columns:
                match_row[f'{key}_score']=fuzzy_match(gold_row.get(key),pred_row.get(key))

            match_data.append(match_row)

    if match_data:
        match_df=pd.DataFrame(match_data)

    else: #create emtpy match dataframe
        match_df=pd.DataFrame(columns=['g','p','scientific_name_gold','scientific_name_pred']+[f'{c}_score' for c in gold_df.columns])

    #only mean over gold columns!
    #match_df['mean_score']=match_df[[f'{k}_score' for k in intersection_keys if k in gold_df.columns]].mean(axis=1)
    #match_df['mean_score']=match_df[[f'{k}_score' for k in gold_df.columns]].mean(axis=1)
    match_df['mean_score']=match_df[[k for k in match_df.columns if k.endswith('_score')]].mean(axis=1)

    return match_df

def get_match(gold_df,pred_df):

    if all([df.attrs['has_meta_row']==True for df in [gold_df,pred_df]]):
        meta_df=cross_match(gold_df[:1],pred_df[:1])
        obs_df=cross_match(gold_df[1:],pred_df[1:])
        has_meta_row=True

    elif all([df.attrs['has_meta_row']==False for df in [gold_df,pred_df]]):
        meta_df=cross_match(gold_df[:0],pred_df[:0])
        obs_df=cross_match(gold_df,pred_df)
        has_meta_row=False

    #scientific_name of bee must match!
    if "scientific_name_score" in obs_df:
        obs_df=obs_df[(obs_df.scientific_name_score>0.9)|(obs_df.scientific_name_score==torch.nan)|(obs_df.scientific_name_score.isna())]

    obs_df.sort_values(by='mean_score',ascending=False,inplace=True)
    obs_df.reset_index(drop=True,inplace=True)
    #return obs_df

    #keep only unique 1:1 mappings
    i=0
    while True:
        if not (obs_df[:i+1]['g'].is_unique) & (obs_df[:i+1]['p'].is_unique):
            obs_df.drop(obs_df.index[i],inplace=True)
            continue
        if i==len(obs_df): break
        if i > 10000: break
        i+=1
    
    column_order=['g','p','scientific_name_gold','scientific_name_pred']+[f'{c}_score' for c in pred_df.columns]+['mean_score']

    if (meta_df.empty) and (obs_df.empty):
        match_df=pd.DataFrame(columns=column_order)
    
    else:
        match_df=pd.concat([
            meta_df if not meta_df.empty else None,
            obs_df if not obs_df.empty else None],
            axis=0,ignore_index=True)
    
        for col in column_order:
            if col not in match_df.columns:
                match_df[col]=torch.nan

    if not has_meta_row:
        match_df.index+=1

    match_df.attrs['has_meta_row']=has_meta_row
    match_df.attrs['gold_is_empty']=gold_df.attrs['is_empty']
    match_df.attrs['pred_is_empty']=pred_df.attrs['is_empty']
    return match_df[column_order]

# SPAN LOCALISATION
def get_observations_span(json_string,verbose=False):
    l0,l1=re.search(r'"observations"\s*:\s*(.*)^(?:(.*?)\s*\}\s*```)',json_string,flags=re.S|re.M).span(1)
    if verbose:
        print(f"get_observations_span: ({l0},{l1})")
    return (l0,l1)

def get_observation_span(idx,json_string,observations_span=None,verbose=False):

    if not isinstance(observations_span,tuple):
        l0,l1=get_observations_span(json_string)
    else:
        l0,l1=observations_span
    
    #that means in meta-space --> return span of meta-attributes
    if idx==0:
        o1,o0,i=l0,0,None
    else:
        #for i,observation in enumerate(re.finditer(r'( *{\s*[^{}]*?\s*\})',json_string[l0:l1])):
        #for i,observation in enumerate(re.finditer(r'{((?:[^{}]|(?R))*)}',json_string[l0:l1])):
        for i,observation in enumerate(re.finditer(r'(\{(?:[^{}]++|(?R))*\})',json_string[l0:l1])):


            if i+1==idx:
                o0,o1=observation.span(1)
                o0+=l0;o1+=l0
                break
    
    if verbose:
        print(f"get_observation_span: i={i} ({o0},{o1})")

    return (o0,o1)

def get_item_span(key='.*?',val='.*?',json_string='',observation_span=None,idx=None,observations_span=None,verbose=False):

    if isinstance(observation_span,tuple):
            o0,o1=observation_span
        
    elif idx:
        o0,o1=get_observation_span(idx,json_string,observations_span)

    else: 
        o0,o1=0,len(json_string)

    if isinstance(val,str): item_pattern=fr'"(?P<key>{key})"\s*:\s*"(?P<val>[^"]*)"'
    elif isinstance(val,bool): item_pattern=fr'"(?P<key>{key})"\s*:\s*(?P<val>true|false)'
    elif isinstance(val,list): item_pattern=fr'"(?P<key>{key})"\s*:\s*(?P<val>\[(?:[^\[\]]|(?P>val))*\])'
    elif isinstance(val,dict): item_pattern=fr'"(?P<key>{key})"\s*:\s*(?P<val>\{{(?:[^{{}}]*|(?P>val))*\}})'
    else: item_pattern=fr'"(?P<key>l{key})"\s*:\s*(?P<val>.*?)[\s,]*$'

    item_span, key_span, value_span = None, None, None

    result=re.search(item_pattern,json_string[o0:o1])
    if result:
        i0,i1=result.span(0)
        k0,k1=result.span(1)
        v0,v1=result.span(2)

        item_span=(i0+o0, i1+o0)
        key_span=(k0+o0, k1+o0)
        value_span=(v0+o0, v1+o0)

        # get_item_span.item_span=(i0+o0, i1+o0)
        # get_item_span.key_span=(k0+o0, k1+o0)
        # get_item_span.value_span=(v0+o0, v1+o0)

    else:
        print(f'Error Itemspan: {locals()}')
        
    if verbose:
        print(f"get_item_span: ({i0+o0},{i1+o0})")

    return item_span,key_span,value_span

# CLASSIFY ERRORS
def get_errors(match_df,gold_df,pred_df):

    if all([df.attrs['has_meta_row']==True for df in [match_df,gold_df,pred_df]]): has_meta_row=True
    elif all([df.attrs['has_meta_row']==False for df in [match_df,gold_df,pred_df]]): has_meta_row=False

    if pred_df.attrs['is_valid']: pred_is_valid=True
    else: pred_is_valid=False

    error_data=[]
    # aggregate attribute errors
    for m,match_row in match_df.iterrows():
        for k,v in match_row.items():
            k=k.replace('_score','')

            #if not isinstance(v,(float,int)): continue
            #if (k.endswith('_score')&(k!='mean_score')&(v<1.0)):

            if k not in pred_df.columns: continue #No value column

            g=int(match_row['g'])
            p=int(match_row['p'])

            gold_val=gold_df.loc[g].get(k)
            pred_val=pred_df.loc[p].get(k)

            if isinstance(pred_val,(dict,list)):
                _pred_val=str(pred_val)
            else:
                _pred_val=pred_val

            error_type='unk'
            is_error=True
            fuzzy_score=0.0

            if pd.isna(gold_val) & pd.isna(_pred_val): 
                error_type='tn_val'
                is_error=False
                fuzzy_score=1.0

            elif pd.notna(gold_val) & pd.notna(_pred_val):
                fuzzy_score=v
                error_type='tp_val'
                is_error=(fuzzy_score<1.0)

            elif k not in gold_df.columns:
                error_type='fp_key'

            elif pd.isna(gold_val) & pd.notna(_pred_val):
                error_type='fp_val'
            
            elif pd.notna(gold_val) & pd.isna(_pred_val):
                error_type='fn_val'

                
            if (g==0) & (p==0): k=f'meta_{k}'

            error_row={
                'm':m,
                'g':g,
                'p':p,
                'key':k,
                'fuzzy_score':fuzzy_score,
                'gold_val':gold_val,
                'pred_val':pred_val,
                'type':error_type,
                'is_error':is_error
            }

            error_data.append(error_row)


    #true positives observations
    for i,tp_row in match_df.iterrows():
        if i==0: continue
        error_row={
            'm':i,
            'g':tp_row['g'],
            'p':tp_row['p'],
            'key':'observations',
            'fuzzy_score':1.0,
            'type': 'tp_obs',
            'is_error':False
        }
        error_data.append(error_row)

    #false positive observations
    false_positives=list(set(pred_df.index)-set(match_df.p)-{0})
    for _,fp_row in pred_df.loc[false_positives].iterrows():
        error_row={
            'm':None,
            'g':None,
            'p':fp_row.name,
            'key':'observations',
            'fuzzy_score':0.0,
            'type': 'fp_obs',
            'is_error':True
        }
        error_data.append(error_row)
    

    #false negative observations
    false_negatives=list(set(gold_df.index)-set(match_df.g)-{0})
    for _,fn_row in gold_df.loc[false_negatives].iterrows():

        error_row={
            'm':None,
            'g':fn_row.name,
            'p':None,
            'key':'observations',
            'fuzzy_score':0.0,
            'type': 'fn_obs',
            'is_error':True
        }
        error_data.append(error_row)

    col_order=["m","g","p","key","fuzzy_score","gold_val","pred_val","type","is_error"]
    errors_df=pd.DataFrame(error_data,columns=col_order)
    errors_df.attrs['has_meta_row']=has_meta_row
    errors_df.attrs['gold_is_empty']=match_df.attrs['gold_is_empty']
    errors_df.attrs['pred_is_empty']=match_df.attrs['pred_is_empty']
    errors_df.attrs['pred_is_valid']=pred_is_valid
    
    return errors_df

def get_sub_spans(gold_val, pred_val,gold_span=None, pred_span=None,verbose=False):

    matcher = difflib.SequenceMatcher(lambda x: x in " \t,.!?", gold_val, pred_val)

    gold_delta,pred_delta=0,0
    if gold_span: gold_delta=gold_span[0]
    if pred_span: pred_delta=pred_span[0]

    gold_sub_spans,pred_sub_spans=[],[]

    for (tag, g1, g2, p1, p2) in matcher.get_opcodes():

        if tag == 'insert': 
            pred_sub_spans.append((p1+pred_delta,p2+pred_delta))

        elif tag == 'delete': 
            gold_sub_spans.append((g1+gold_delta,g2+gold_delta))

        elif tag == 'replace': 
            gold_sub_spans.append((g1+gold_delta,g2+gold_delta))
            pred_sub_spans.append((p1+pred_delta,p2+pred_delta))

        if verbose: print(f"{tag}\t{gold_val[g1:g2].strip()} --> {pred_val[p1:p2].strip()}")
    
    return gold_sub_spans,pred_sub_spans

def get_error_spans(errors_df,gold_completion,pred_completion,verbose=False):

    errors_df[['gold_span','pred_span','gold_sub_spans','pred_sub_spans']]=None,None,None,None

    #for error_row in errors_df.itertuples():
    for i,error_row in errors_df.iterrows():
        #print(error_row.Index)

        if error_row['type'] in ['tp_obs','fp_obs']:
            obs_span=get_observation_span(error_row['p'],pred_completion,verbose=verbose)
            errors_df.at[i,'pred_span']=[obs_span]
            errors_df.at[i,'pred_sub_spans']=[obs_span]
            errors_df.at[i,'pred_val']=pred_completion[obs_span[0]:obs_span[1]]

        if error_row['type']  in ['tp_obs','fn_obs']:
            obs_span=get_observation_span(error_row['g'],gold_completion,verbose=verbose)
            errors_df.at[i,'gold_span']=[obs_span]
            errors_df.at[i,'gold_sub_spans']=[obs_span]
            errors_df.at[i,'gold_val']=gold_completion[obs_span[0]:obs_span[1]]

        

        elif (error_row['type'] in ['tp_val']) and (error_row['is_error']):
            _,_,gold_value_span=get_item_span(error_row['key'],val=error_row['gold_val'],json_string=gold_completion,idx=error_row['g'],verbose=verbose)
            errors_df.at[i,'gold_span']=[gold_value_span]

            _,_,pred_value_span=get_item_span(error_row['key'],val=error_row['pred_val'],json_string=pred_completion,idx=error_row['p'],verbose=verbose)
            errors_df.at[i,'pred_span']=[pred_value_span]

            gold_sub_value_spans,pred_sub_value_spans=get_sub_spans(error_row['gold_val'], error_row['pred_val'], gold_value_span, pred_value_span, verbose=verbose)
            errors_df.at[i,'gold_sub_spans']=gold_sub_value_spans
            errors_df.at[i,'pred_sub_spans']=pred_sub_value_spans


        elif error_row['type'] in ['fp_val']:
            _,_,pred_value_span=get_item_span(error_row['key'],val=error_row['pred_val'],json_string=pred_completion,idx=error_row['p'],verbose=verbose)
            errors_df.at[i,'pred_span']=[pred_value_span]
            errors_df.at[i,'pred_sub_spans']=[pred_value_span]

        
        elif error_row['type'] in ['fn_val']:
            _,_,gold_value_span=get_item_span(error_row['key'],val=error_row['gold_val'],json_string=gold_completion,idx=error_row['g'],verbose=verbose)
            errors_df.at[i,'gold_span']=[gold_value_span]
            errors_df.at[i,'gold_sub_spans']=[gold_value_span]

        elif error_row['type'] in ['fp_key']:
            _,pred_key_span,_=get_item_span(error_row['key'],val=error_row['pred_val'],json_string=pred_completion,idx=error_row['p'],verbose=verbose)
            errors_df.at[i,'pred_span']=[pred_key_span]
            errors_df.at[i,'pred_sub_spans']=[pred_key_span]
  
    return errors_df


ERROR_HIGHLIGHT_COLORS={
    'tp_val':'#DA70D6',
    'fp_key':'#DE3163',
    "fp_val":"#DA70D6",
    "fn_val":"#DA70D6",
    "fp_obs":"#DA70D6",
    "fn_obs":"#DA70D6",
    "total_val":"#FF0000",
    "partial_val":"#FA8072",   
}

def get_red_hexgradient(score):
    r = 255
    g = b = int(255 * score)
    return f"#{r:02X}{g:02X}{b:02X}"

def get_char_highlighting(plain_text,errors_df,col='pred'):

    if errors_df.empty:
        return plain_text

    plain_text=str(plain_text)
    fancy_text=""
    last_i1=0

    errors_df=errors_df[errors_df.is_error]
    for _,error_row in errors_df.sort_values(by=f"{col}_span").iterrows():

        color=ERROR_HIGHLIGHT_COLORS[error_row['type']]
        spans=error_row[f"{col}_sub_spans"]

        if not spans: continue

        for (i0,i1) in spans:
            fancy_text+=plain_text[last_i1:i0]+f'<span style="color:{color}">'+plain_text[i0:i1]+'</span>'
            last_i1=i1

    fancy_text+=plain_text[last_i1:]
    return fancy_text


# CHARS to TOKENS MAPPING
def check_span_overlap(span,target_span):
    if not span: return False
    return span[0]<target_span[1] and span[1] > target_span[0]

def check_spans_overlap(spans,target_span):
    if not spans: return [False]
    return list(map(lambda x: check_span_overlap(x,target_span), spans))

def get_tokens(completion,errors_df,col="pred",tokenizer=None):

    col_order=['chars_span', 'chars', 'id', 'error_idx','error_sub_grp','error_key', 'error_type','error_span', 'error_sub_spans']
    errors_df=errors_df[errors_df.is_error]

    if errors_df.empty:
        return pd.DataFrame(columns=col_order)

    completion=str(completion)
    encoded = tokenizer(completion, return_offsets_mapping=True,add_special_tokens=False)

    token_data=[]
    for i,(c0,c1) in enumerate(encoded["offset_mapping"]):

        token_row={
            "chars_span":(c0,c1),
            "chars":completion[c0:c1],
            "id": encoded["input_ids"][i],
            "error_idx":-1,
            "error_key":None,
            "error_type": None,
            "error_span":None,
            "error_sub_spans":None
        }

        #get corresponding error_rows
        error_rows=errors_df[errors_df.apply(lambda x: any(check_spans_overlap(x.get(f"{col}_span"),(c0,c1))),axis=1)]

        if not error_rows.empty:

            if len(error_rows)>1: print("ERROR! more than one corresponding error_row!!")

            error_row=error_rows.iloc[0]
            token_row["error_idx"]=error_row.name
            error_row=error_row.to_dict()

            token_row["error_span"]=error_row[f"{col}_span"]

            sub_span_matches=check_spans_overlap(error_row[f"{col}_sub_spans"],(c0,c1))
            sub_spans=[span for span,match in zip(error_row[f"{col}_sub_spans"],sub_span_matches) if match]
            sub_spans= torch.nan if not sub_spans else sub_spans
            token_row["error_sub_spans"]=sub_spans

            #take over values from corresponding error_row
            error_keys=["key","type"]
            for key in error_keys:
                token_row[f"error_{key}"]=error_row[key]

        token_data.append(token_row)

    tokens_df=pd.DataFrame(token_data)
    
    #if 'error_sub_spans' in tokens_df:
    tokens_df['error_sub_grp'], _=pd.factorize(tokens_df['error_sub_spans'].apply(lambda x: str(x) if isinstance(x,list) else x))

    
    return tokens_df[col_order]


# ERROR HIGHLIGHTING (Tokens)
def set_group_value(group,key="html_intro_tag",value="first",idx=0):

    if group.name==-1:
        group[key]=None

    else:
        group.loc[group.index[idx], key] = value
    
    return group

def get_token_highlighting(tokens_df):

    tokens_df=tokens_df.drop_duplicates(subset=['chars_span'])
    tokens_df=tokens_df.groupby('error_sub_grp').apply(lambda x: set_group_value(x,key='html_intro_tag',value='<span style="color:#DA70D6">',idx=0),include_groups=False).reset_index(level=0, drop=False).sort_index()
    tokens_df=tokens_df.groupby('error_sub_grp').apply(lambda x: set_group_value(x,key='html_outro_tag',value='</span>',idx=-1),include_groups=False).reset_index(level=0, drop=False).sort_index()
    tokens_df[['html_intro_tag','html_outro_tag']]=tokens_df[['html_intro_tag','html_outro_tag']].fillna('')


    fancy_text=""
    for i,token_row in tokens_df.iterrows():
        
        prefix,suffix="",""
        #if i%2 == 0: prefix,suffix="<u>","</u>" #mark single tokens
        fancy_text+=prefix+token_row['html_intro_tag']+token_row['chars']+token_row['html_outro_tag']+suffix
    
    return fancy_text

#### Metric: Confusion with Fuzzy-Loss
def precision(tp=None,fp=None,fn=None,tn=None,tp_fuzzy=None,use_fuzzy=True,fast_return=None):

    if use_fuzzy and pd.notna(tp_fuzzy):
        precision.label='P_fuzzy'
        _tp=tp_fuzzy
    else:
        precision.label='P'
        _tp=tp
    
    if fast_return=='best': return 1.0
    elif fast_return=='worst': return 0.0

    if not all([pd.notna(k) for k in [_tp,tp,fp]]): return None
    if _tp==0: return 0.0

    return _tp/(tp+fp)

def recall(tp=None,fp=None,fn=None,tn=None,tp_fuzzy=None,use_fuzzy=True,fast_return=None):
    
    if use_fuzzy and pd.notna(tp_fuzzy):
        recall.label='R_fuzzy'
        _tp=tp_fuzzy
    else:
        recall.label='R'
        _tp=tp
    
    if fast_return=='best': return 1.0
    elif fast_return=='worst': return 0.0

    if not all([pd.notna(k) for k in [_tp,tp,fn]]): return None
    if _tp==0: return 0.0

    return _tp/(tp+fn)

def accuracy(tp=None,fp=None,fn=None,tn=None,tp_fuzzy=None,use_fuzzy=True,fast_return=None):
    
    if use_fuzzy and pd.notna(tp_fuzzy):
        accuracy.label='ACC_fuzzy'
        _tp=tp_fuzzy
    else:
        accuracy.label='ACC'
        _tp=tp
    
    if fast_return=='best': return 1.0
    elif fast_return=='worst': return 0.0
    
    if not all([pd.notna(k) for k in [_tp,tp,fp,fn,tn]]): return None

    numerator=_tp+tn
    if numerator==0: return 0.0

    denominator=tp+fn+fp+tn
    return numerator/denominator

def balanced_error_rate(tp=None,fp=None,fn=None,tn=None,tp_fuzzy=None,use_fuzzy=True,fast_return=None):

    if use_fuzzy and pd.notna(tp_fuzzy):
        balanced_error_rate.label='BER_fuzzy'
        _tp=tp_fuzzy
    else:
        balanced_error_rate.label='BER'
        _tp=tp

    if fast_return=='best': return 0.0
    elif fast_return=='worst': return 1.0

    if not all([pd.notna(k) for k in [_tp,tp,fp,fn,tn]]): return None

    ber=1.0
    if (_tp+fn)!=0: ber-=0.5*(_tp/(tp+fn))
    if (tn+fp)!=0: ber-=0.5*(tn/(tn+fp))

    return ber

def f1_score(tp=None,fp=None,fn=None,tn=None,tp_fuzzy=None,use_fuzzy=True,fast_return=None):

    if use_fuzzy and pd.notna(tp_fuzzy):
        f1_score.label='F1_fuzzy'
        _tp=tp_fuzzy
    else:
        f1_score.label='F1'
        _tp=tp

    if fast_return=='best': return 1.0
    elif fast_return=='worst':return 0.0

    if _tp==0: return 0.0

    return (_tp/(tp+0.5*(fp+fn)))

def get_conf_dict(errors_df):

    conf_dict=dict()
    conf_dict['has_meta_row']=errors_df.attrs['has_meta_row']

    conf_dict['gold_is_empty']=errors_df.attrs['gold_is_empty']
    conf_dict['pred_is_empty']=errors_df.attrs['pred_is_empty']


    conf_dict['pred_is_valid']=errors_df.attrs['pred_is_valid']

    #OBSERVATIONS
    selected_df=errors_df[errors_df.key=='observations']
    conf_dict['obs']={
        'tp':len(selected_df[selected_df.type.isin(['tp_obs'])]),
        #'tp_fuzzy':selected_df[selected_df.type.isin(['tp_obs'])]['fuzzy_score'].sum(),
        'fp':len(selected_df[selected_df.type.isin(['fp_obs'])]),
        'fn':len(selected_df[selected_df.type.isin(['fn_obs'])]),
        'tn':len(selected_df[selected_df.type.isin(['tn_obs'])]),
    }

    #META ATTRIBUTES
    if errors_df.attrs['has_meta_row']:
        selected_df=errors_df[(errors_df.g==0)&(errors_df.p==0)]
        conf_dict['meta_vals']={
            'tp':len(selected_df[selected_df.type.isin(['tp_val'])]),
            'tp_fuzzy':selected_df[selected_df.type.isin(['tp_val'])]['fuzzy_score'].sum(),
            'fp':len(selected_df[selected_df.type.isin(['fp_val'])]),
            'fn':len(selected_df[selected_df.type.isin(['fn_val'])]),
            'tn':len(selected_df[selected_df.type.isin(['tn_val'])]),
        }

    #OBSERVATION VALUES
    selected_df=errors_df[(errors_df.g!=0)&(errors_df.p!=0)]
    conf_dict['obs_vals']={
        'tp':len(selected_df[selected_df.type.isin(['tp_val'])]),
        'tp_fuzzy':selected_df[selected_df.type.isin(['tp_val'])]['fuzzy_score'].sum(),
        'fp':len(selected_df[selected_df.type.isin(['fp_val'])]),
        'fn':len(selected_df[selected_df.type.isin(['fn_val'])]),
        'tn':len(selected_df[selected_df.type.isin(['tn_val'])]),
    }

    return conf_dict

def get_eval_dict(conf_dict):
    metric_funcs=[precision,recall,f1_score,accuracy,balanced_error_rate]

    eval_dict=dict()
    eval_dict['has_meta_row']=conf_dict['has_meta_row']

    eval_dict['gold_is_empty']=conf_dict['gold_is_empty']
    eval_dict['pred_is_empty']=conf_dict['pred_is_empty']

    eval_dict['pred_is_valid']=conf_dict['pred_is_valid']
    

    fast_return=None
    if conf_dict['pred_is_valid']==False: fast_return='worst'
    elif conf_dict['gold_is_empty'] and conf_dict['pred_is_empty']: fast_return='best'
    elif conf_dict['gold_is_empty'] != conf_dict['pred_is_empty']: fast_return='worst'

    for scope in conf_dict.keys():
        if not isinstance(conf_dict[scope],dict): continue

        eval_dict[scope]=dict()
        for metric_func in metric_funcs:
            
            score=metric_func(**conf_dict[scope],fast_return=fast_return)
            eval_dict[scope][metric_func.label]=score
    
    #eval_dict['fuzzy_score']=(eval_dict['obs']['F1']+eval_dict['obs_vals']['F1_fuzzy'])/2
    eval_dict['fuzzy_score']=eval_dict['obs']['F1']*eval_dict['obs_vals']['F1_fuzzy']
    
    return eval_dict
