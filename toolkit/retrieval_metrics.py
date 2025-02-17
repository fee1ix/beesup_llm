import pandas as pd

def precision(tp=None,fp=None,fn=None,tn=None,tp_fuzzy=None,use_fuzzy=True,fast_return=None, **kwargs):

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

def recall(tp=None,fp=None,fn=None,tn=None,tp_fuzzy=None,use_fuzzy=True,fast_return=None, **kwargs):
    
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

def accuracy(tp=None,fp=None,fn=None,tn=None,tp_fuzzy=None,use_fuzzy=True,fast_return=None, **kwargs):
    
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

def balanced_error_rate(tp=None,fp=None,fn=None,tn=None,tp_fuzzy=None,use_fuzzy=True,fast_return=None, **kwargs):

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
