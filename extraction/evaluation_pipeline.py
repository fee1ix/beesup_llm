from .extraction_pipeline import *

from .evaluation_utils import *

class EvaluationSample(ExtractionSample):

    def __init__(self, gold_completion=None, pred_completion=None, **kwargs):
        super().__init__(pred_completion, **kwargs)

        self.gold_completion=gold_completion

        assert hasattr(self, 'gold_completion'), "missing 'gold_completion'"

        self.parse_json(prefix='gold', exclude_none=True)
        self.parse_df(prefix='gold')

        self.evaluate()

    def evaluate(self):

        gold_df=self.gold_df
        pred_df=self.pred_df
        
        match_df=get_match(gold_df,pred_df)
        self.match_df=match_df

        errors_df=get_errors(match_df,gold_df,pred_df)
        self.errors_df=errors_df

        conf_dict=get_conf_dict(errors_df)
        self.conf_dict=conf_dict

        eval_dict=get_eval_dict(conf_dict)
        self.eval_dict=eval_dict

        self.total_score=eval_dict['total_score']
    
    def evaluate_raw(self):

        self.parse_df(prefix='gold', create_meta_row=True)
        self.parse_df(prefix='pred', create_meta_row=True)

        raw_gold_df=self.raw_gold_df
        raw_pred_df=self.raw_pred_df

        raw_match_df=get_match(raw_gold_df,raw_pred_df)
        self.raw_match_df=raw_match_df

        raw_errors_df=get_errors(raw_match_df,raw_gold_df,raw_pred_df)
        self.raw_errors_df=raw_errors_df

        raw_conf_dict=get_conf_dict(raw_errors_df)
        self.raw_conf_dict=raw_conf_dict

        raw_eval_dict=get_eval_dict(raw_conf_dict)
        self.raw_eval_dict=raw_eval_dict

        self.raw_total_score=raw_eval_dict['total_score']

    def load_highlighting(self):

        self.evaluate_raw()
        self.raw_errors_df=get_error_spans(self.raw_errors_df,self.gold_completion,self.pred_completion,verbose=False)
        self.gold_highlighting=get_char_highlighting(self.gold_completion,self.raw_errors_df,col='gold')
        self.pred_highlighting=get_char_highlighting(self.pred_completion,self.raw_errors_df,col='pred')
        
    def get_errors_df(self):

        errors_df=self.errors_df.copy()
        errors_df['pred_val'] = errors_df['pred_val'].astype('object')
        errors_df['gold_val'] = errors_df['gold_val'].astype('object')

        #errors_df=errors_df[errors_df.is_error==True]
        errors_df['idx']=None
        for i,error_row in errors_df.iterrows():
            if error_row.type in ['fp_obs', 'tp_obs']:
                errors_df.at[i,'pred_val']=self.pred_df.iloc[int(error_row.p)].scientific_name
            
            if error_row.type in ['fn_obs', 'tp_obs']:
                errors_df.at[i,'gold_val']=self.gold_df.iloc[int(error_row.g)].scientific_name

        return errors_df

    
    def __repr__(self):
        if not hasattr(self, 'gold_highlighting'): self.load_highlighting()
        if not hasattr(self, 'pred_highlighting'): self.load_highlighting()

        from ..toolkit.visualization import print_multicol
        print_multicol(["<h2>GOLD-LABEL</h2>",f"<h2>PREDICTION (total_score={self.total_score:.3f})</h2>"])
        print_multicol([self.gold_highlighting,self.pred_highlighting])
        return ''


class EvaluationPipeline(ExtractionPipeline):
    type='evaluation_pipeline'

    def __init__(self, ref=None, **kwargs):

        super().__init__(ref,  **kwargs)

        self._config_key_order.extend([])
        self._config_keys_to_exclude.extend([])

        self._default_config=dict()
        self.update_config(self._default_config, overwrite_if_conflict=False)

    
    def get_eval(self, gold_completion=None, pred_completion=None, **kwargs):
        sample=EvaluationSample(gold_completion=gold_completion, pred_completion=pred_completion, **kwargs)
        return sample.eval_dict
    
    def get_errors_df(self, df, **kwargs):
        df=df.copy()
        assert 'gold_completion' in df.columns, "missing 'gold_completion' column"
        assert 'pred_completion' in df.columns, "missing 'pred_completion' column"

        errors_df=pd.DataFrame()
        errors_df['i']=None
        for i,row in df.iterrows():
            sample=EvaluationSample(**row)
            _errors_df=sample.get_errors_df()
            
            if _errors_df.empty: continue
            _errors_df['global_step']=row.get('global_step',None)
            _errors_df['epoch']=row.get('epoch',None)

            _errors_df['i']=i
            _errors_df.dropna(axis=1, how='all')
            
            errors_df=pd.concat([errors_df,_errors_df], ignore_index=True)


        return errors_df

    
    def get_eval_df(self, df, **kwargs):
        df=df.copy()
        assert 'gold_completion' in df.columns, "missing 'gold_completion' column"
        assert 'pred_completion' in df.columns, "missing 'pred_completion' column"

        df['eval_dict']=None
        for i,row in df.iterrows():
            sample=EvaluationSample(**row)
            df.at[i,'eval_dict']=sample.eval_dict

        return df
  

    def __call__(self, the_input, condense=False, **kwargs):

        if isinstance(the_input,pd.DataFrame):
            return self.get_eval_df(the_input, **kwargs)
        
        else:
            return EvaluationSample(**the_input, **kwargs)









