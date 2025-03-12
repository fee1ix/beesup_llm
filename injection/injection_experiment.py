import copy
import pandas as pd

from beesup_llm import _isinstance
from beesup_llm.finetuning_experiment import *
from beesup_llm.injection.rag import RAGPipeline
from beesup_llm.injection.taxomizing import Taxomizer
from beesup_llm.injection.injection_evaluation import *

class InjectionCallback(EvaluatorCallback):

    def __init__(self, *args, rag_pipe: RAGPipeline=None, **kwargs):
        super().__init__(*args, **kwargs)

        if _isinstance(rag_pipe, RAGPipeline):
            self.rag_pipe = rag_pipe
        
        evaluator_tag=self.evaluator.__class__.__name__.lower()[:3]+f":{self.evaluator.id}"
        rag_pipe_tag='rag' if hasattr(self, 'rag_pipe') else ''
        self.object_str=f"{evaluator_tag}-{rag_pipe_tag}"

    def on_epoch_end(self, args=None, state=None, control=None, **kwargs):
        super().on_epoch_end(args, state, control, **kwargs)
        
        if not self.evaluator.is_eval_epoch(state.epoch):
            self.experiment.logger.info(f"{self.state_str} Skip, because {state.epoch} not in {self.evaluator.eval_epochs=}")
            return #skip evaluation if not specified as eval epoch

        model=kwargs['model']
        model.eval()

        pipe_df=self.evaluator.eval_df.iloc[:4].copy()
        if hasattr(self,'rag_pipe'):
            kwargs.update({k: getattr(self.rag_pipe, k, None) for k in ['chunk_txt_key', 'query_txt_key']})
            self.rag_pipe.add_ranking_df(pipe_df)
            self.rag_pipe.add_briefing_df(pipe_df)
        
    
        pipe_df=self.evaluator.get_pipe_df(pipe_df, toc=self.experiment.toc, **kwargs) #add prompt messages, if rag

        self.experiment.logger.info(f"{self.state_str} {self.object_str} {kwargs=}")
        callback_df=self.evaluator(
            llm_pipe=self.experiment.llm_pipe.__class__(model=model),
            pipe_df=pipe_df,
            **kwargs)

        self.save_df(callback_df, state)

class InjectionExperiment(FinetuningExperiment):

    @classmethod
    def fit_llm_pipe(cls, llm_pipe: LLMPipeline, **kwargs) -> LLMPipeline:
        llm_pipe=super().fit_llm_pipe(llm_pipe, **kwargs)
        return llm_pipe

    def __init__(self, *args, taxomizer: Taxomizer=None, rag_pipe: RAGPipeline = None, **kwargs):

        self.do_eval_base_model_rag=kwargs.get('do_eval_base_model_rag', False)
        self.do_eval_lora_model_rag=kwargs.get('do_eval_lora_model_rag', False)

        super().__init__(*args, **kwargs)

        if hasattr(self, 'labh'):
            rag_pipe=self.labh.handle_object(locals(),'rag_pipe')

        if _isinstance(rag_pipe, RAGPipeline):
            self.rag_pipe=rag_pipe
            self.rag_pipe.llm_pipe=getattr(self, 'llm_pipe', None)
            self.rag_pipe.add_limiter_features()
        
        self.toc=None #extract toc from injection data
        if isinstance(self.data_df, pd.DataFrame):
            if not self.data_df.empty:
                if 'toc' in self.data_df.columns:
                    self.toc=self.data_df['toc'].iloc[0]
    

    def run(self, **kwargs) -> None:

        self.run_entry(**kwargs)

        if self.do_eval_base_model_rag:
            self.evaluate_base_model(callback_class=InjectionCallback, rag_pipe=self.rag_pipe, **kwargs)
        
        if self.do_eval_base_model:
            self.evaluate_base_model(callback_class=InjectionCallback, **kwargs)
        
        if (self.do_eval_lora_model or self.do_train):
            self.load_trainer(**kwargs)

        if self.do_eval_lora_model_rag:
            for evaluator in self.evaluators:
                evaluator_callback=InjectionCallback(self, evaluator, rag_pipe=self.rag_pipe, **kwargs)
                self.trainer.add_callback(evaluator_callback)

        if self.do_eval_lora_model:
            for evaluator in self.evaluators:
                evaluator_callback=InjectionCallback(self, evaluator, **kwargs)
                self.trainer.add_callback(evaluator_callback)

        if self.do_train:
            for trainer_callback in [MCECallback(self), HistoryCallback(self)]:
                self.trainer.add_callback(trainer_callback)
            self.train(**kwargs)

        self.run_exit(**kwargs)
        return
        



        






