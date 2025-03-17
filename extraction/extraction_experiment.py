import copy
import pandas as pd

from typing import Union
from beesup_llm import _isinstance
from beesup_llm.finetuning_experiment import *
from beesup_llm.llm_evaluation import LLMEvaluator	
from beesup_llm.extraction.extraction_pipeline import ExtractionPipeline

class ExtractionEvaluator(LLMEvaluator):
    def __init__(self, *args, extraction_pipe: ExtractionPipeline=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if hasattr(self, 'labh'):
            extraction_pipe = self.labh.handle_object(locals(), 'extraction_pipe')

        if _isinstance(extraction_pipe, ExtractionPipeline):
            self.extraction_pipe=extraction_pipe
            self.extraction_pipe.llm_pipe=getattr(self, 'llm_pipe', None) or getattr(self.extraction_pipe, 'llm_pipe', None)
            
            if not hasattr(self.extraction_pipe, 'llm_pipe'):
                raise ValueError("neighter llm_pipe in args nor in extraction_pipe")
        
        self.extraction_pipe.add_prompt_messages(self.eval_df)
    
    def __call__(self, **kwargs) -> pd.DataFrame:

        pipe_df=self.eval_df.iloc[:2].copy()
        self.add_pred_completion(pipe_df, **kwargs)

        self.extraction_pipe.add_pred_dict(pipe_df, **kwargs)
        self.extraction_pipe.add_eval_dict(pipe_df, **kwargs)

        return pipe_df
  
class ExtractionCallback(EvaluatorCallback):

    def on_epoch_end(self, args=None, state=None, control=None, **kwargs) -> None:
        super().on_epoch_end(args, state, control, **kwargs)

        if not self.evaluator.is_eval_epoch(state.epoch):
            self.experiment.logger.info(f"{self.state_tag} Skip, because {state.epoch} not in {self.evaluator.eval_epochs=}")
            return #skip evaluation if not specified as eval epoch

        model=kwargs.pop('model')
        model.eval()

        callback_df=self.evaluator(
            llm_pipe=self.experiment.llm_pipe.__class__(model=model),
            **kwargs)

        self.save_df(callback_df, state)
        return

class ExtractionExperiment(FinetuningExperiment):

    @classmethod
    def fit_llm_pipe(cls, llm_pipe: LLMPipeline, **kwargs) -> LLMPipeline:
        llm_pipe=super().fit_llm_pipe(llm_pipe, **kwargs)
        return llm_pipe

    def __init__(self, *args, extraction_pipe: ExtractionPipeline = None, **kwargs) -> None:

        self.eval_epochs=kwargs.get('eval_epochs', [])
        super().__init__(*args, **kwargs)

        assert self.evaluators == [], "evaluators should be empty" #evaluator is derrived from extraction_pipe in run()

        if hasattr(self, 'labh'):
            extraction_pipe=self.labh.handle_object(locals(),'extraction_pipe')

        if _isinstance(extraction_pipe, ExtractionPipeline):
            self.extraction_pipe=extraction_pipe
            self.extraction_pipe.llm_pipe=getattr(self, 'llm_pipe')
               
    def load_data(self, **kwargs) -> None:

        assert hasattr(self, 'data_df'), "data_df is missing"
        assert 'split' in self.data_df.columns, "split column is missing"
        assert hasattr(self, 'llm_pipe'), "llm_pipe is missing"

        assert 'report_passage' in self.data_df.columns, "missing 'report_passage' column"
        assert 'gold_completion' in self.data_df.columns, "missing 'gold_completion' column"
        
        #prepare training data
        train_df=self.data_df[self.data_df['split']=='train'].reset_index(drop=True).copy()
        self.extraction_pipe.add_prompt_messages(train_df, **kwargs)
        train_df['gold_message']=train_df['gold_completion'].apply(lambda x: [{'role':'assistant','content': x}])

        eval_df=self.data_df[self.data_df['split']=='eval'].reset_index(drop=True).copy()
        self.eval_df=eval_df.copy()

        self.train_ds, self.eval_ds = None, None
        if not train_df.empty:
            self.llm_pipe.load_tokenizer('training')
            self.train_ds=Dataset.from_list(train_df.apply(lambda x: prepare_sample_for_chat_finetuning(x, self.llm_pipe.get_tokenizer('training'),**kwargs), axis=1).to_list())
        
        return

    def run(self, **kwargs) -> None:
        self.run_entry(**kwargs)

        evaluators=[
            ExtractionEvaluator(
                eval_df=self.eval_df,
                llm_pipe=self.llm_pipe,
                eval_epochs=self.eval_epochs,
                extraction_pipe=self.extraction_pipe,
                labh=None,
                **kwargs)]

        if self.do_eval_base_model:
            self.evaluate_base_model(evaluators=evaluators, callback_class=ExtractionCallback,**kwargs)
        
        if (self.do_eval_lora_model or self.do_train):
            self.load_trainer(**kwargs)

        if self.do_eval_lora_model:
            for evaluator in evaluators:
                evaluator_callback=ExtractionCallback(experiment=self, evaluator=evaluator, **kwargs)
                self.trainer.add_callback(evaluator_callback)
        
        if self.do_train:
            for trainer_callback in [MCECallback(self), HistoryCallback(self)]:
                self.trainer.add_callback(trainer_callback)
            self.train(**kwargs)
        
        self.run_exit(**kwargs)
        return
    
    
