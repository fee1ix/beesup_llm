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

    
    def load_highlighting(self):

        self.errors_df=get_error_spans(self.errors_df,self.gold_completion,self.pred_completion,verbose=False)

        self.gold_highlighting=get_char_highlighting(self.gold_completion,self.errors_df,col='gold')
        self.pred_highlighting=get_char_highlighting(self.pred_completion,self.errors_df,col='pred')
        
    def __repr__(self):
        if not hasattr(self, 'gold_highlighting'): self.load_highlighting()
        if not hasattr(self, 'pred_highlighting'): self.load_highlighting()

        from ..toolkit.visualization import print_multicol
        print_multicol(["<h2>TARGET</h2>",f"<h2>PREDICTION (total_score={self.total_score:.3f})</h2>"])
        print_multicol([self.gold_highlighting,self.pred_highlighting])





    
    


class EvaluationPipeline(ExtractionPipeline):
    type='evaluation_pipeline'

    def __init__(self, ref=None, **kwargs):

        super().__init__(ref,  **kwargs)

        self._config_key_order.extend([])
        self._config_keys_to_exclude.extend([])

        self._default_config=dict()
        self.update_attributes(self._default_config, overwrite=False)

    
    def get_eval(self, gold_completion=None, pred_completion=None, **kwargs):
        sample=EvaluationSample(gold_completion=gold_completion, pred_completion=pred_completion, **kwargs)
        return sample.eval_dict
    
    def get_eval_df(self, df, **kwargs):

        assert 'gold_completion' in df.columns, "missing 'gold_completion' column"
        assert 'pred_completion' in df.columns, "missing 'pred_completion' column"

        for i,row in df.iterrows():
            sample=EvaluationSample(**row)
            df.at[i,'eval_dict']=sample.eval_dict

        return df
    




    

    # def __call__(self, pred_completion=None, gold_completion=None, **kwargs):


    #     return outputs
    
    # def call_by_row(self, row):




    # def call_by_dict(self, the_dict)









