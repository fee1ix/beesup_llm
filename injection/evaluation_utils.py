import json
from rapidfuzz import fuzz
import pandas as pd
LETTERS='ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def clean_completion(completion):
    completion=completion.replace('<|eot_id|>','')
    return completion

def get_chat_completion(sample, llm_pipe):
    completion=llm_pipe(sample.prompt, use_chatformat=True, stream=False, stop_strings=['\n'])[0]['generated_text']
    completion=clean_completion(completion)
    return completion


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

def f1_score(tp=None,fp=None,fn=None,tn=None,tp_fuzzy=None,use_fuzzy=True,fast_return=None, **kwargs):

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