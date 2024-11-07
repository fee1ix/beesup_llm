from beesup_llm import *
from ..toolkit.setup_utils import *
from ..toolkit.visualization import * 



from beesup_llm.pipeline import BasePipeline
from beesup_llm.model import *

from beesup_llm.pipeline.extraction_utils import *

class ExtractionPipeline(BasePipeline):
    



class ExtractionEvalSample(object):

    def __init__(self, row, tokenizer, **kwargs):

        self.tokenizer=tokenizer # needed for token highlighting

        if isinstance(row,dict):
            for k,val in row.items():
                setattr(self,k,val)
        
        else: # it should be a DataFrame Row
            self.index=row.name
            for col,val in row.items():
                setattr(self,col,val)
        
        self.parse()
        self.process()
        self.evaluate()

    def parse(self):
        self.raw_gold_df,self.tab_gold_df=parse_completion(self.gold_completion,verbose=False)
        self.raw_pred_df,self.tab_pred_df=parse_completion(self.pred_completion,verbose=False)
        self.pred_is_valid=parse_completion.is_valid

    def process(self):
        self.raw_match_df=get_match(self.raw_gold_df,self.raw_pred_df)
        self.tab_match_df=get_match(self.tab_gold_df,self.tab_pred_df)

        #get error list
        self.raw_errors_df=get_errors(self.raw_match_df,self.raw_gold_df,self.raw_pred_df)
        self.tab_errors_df=get_errors(self.tab_match_df,self.tab_gold_df,self.tab_pred_df)

    def evaluate(self):
        self.raw_conf_dict=get_conf_dict(self.raw_errors_df)
        self.tab_conf_dict=get_conf_dict(self.tab_errors_df)

        self.raw_eval_dict=get_eval_dict(self.raw_conf_dict)
        self.tab_eval_dict=get_eval_dict(self.tab_conf_dict)

        self.fuzzy_score=self.tab_eval_dict['fuzzy_score']  

    def compute_highlighting(self,highlighting='chars'):

        self.raw_errors_df=get_error_spans(self.raw_errors_df,self.gold_completion,self.pred_completion,verbose=False)
        
        if highlighting=='tokens':
            self.gold_tokens_df=get_tokens(self.gold_completion,self.raw_errors_df,col='gold')
            self.pred_tokens_df=get_tokens(self.pred_completion,self.raw_errors_df,col='pred')
            
            self.gold_completion_highlighted=get_token_highlighting(self.gold_tokens_df)
            self.pred_completion_highlighted=get_token_highlighting(self.pred_tokens_df)
        
        elif highlighting=='chars':
            self.gold_completion_highlighted=get_char_highlighting(self.gold_completion,self.raw_errors_df,col='gold')
            self.pred_completion_highlighted=get_char_highlighting(self.pred_completion,self.raw_errors_df,col='pred')

    def print_eval(self,highlighting='chars',include_conf=False):

        print_multicol(["<h2>TARGET</h2>",f"<h2>PREDICTION (fuzzy_score={self.fuzzy_score:.3f})</h2>"])
        self.compute_highlighting(highlighting=highlighting)

        print_multicol([self.gold_completion_highlighted,self.pred_completion_highlighted])
        print_multicol(["<h4>Confusion</h4>","<h4>Confusion-Evaluation</h4>"])
        if include_conf:
            print_multicol([json.dumps(self.tab_conf_dict,indent=2),json.dumps(self.tab_eval_dict,indent=2)])
       
class ExtractionEvalPipeline(BasePipeline):

    type='extraction_eval'

    def __init__(self, ref=None, dataset_ref=None, model_ref=None, **kwargs):
        super().__init__(ref, dataset_ref, **kwargs)
        self._config_key_order.extend([])
        self._config_keys_to_exclude.extend([])

        self._default_config=dict()
        self.update_attributes(self._default_config, overwrite=False)

        if model_ref is not None:

            model=BaseModelWrap.from_ref(model_ref)
            self.gen_model_config=model.get_config()
            self.tokenizer=model.get_inference_tokenizer()
    

    #def evaluate(self):

        





    

    


    

