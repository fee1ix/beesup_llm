import copy
import pandas as pd

from beesup_llm import _isinstance
from beesup_llm.finetuning_experiment import FinetuningExperiment, PredictionCallback, EvaluatorCallback
from beesup_llm.injection.rag import RAGPipeline
from beesup_llm.injection.injection_evaluation import InjectionEvaluator, MCQEvaluator, QDQEvaluator, FFQEvaluator

class InjectionCallback(EvaluatorCallback):

    def __init__(self, *args, rag_pipe: RAGPipeline=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.toc = None
        if hasattr(self.experiment,'taxomizer'):
            self.toc=self.experiment.taxomizer.toc
        self.experiment.logger.info(f"{self.toc=}")

        self.rag_pipe = rag_pipe
        self.experiment.logger.info(f"{self.rag_pipe=}")

    def get_df_name(self, state):

        evaluator_tag=self.evaluator.__class__.__name__.lower()[:3]+f":{self.evaluator.id}"
        rag_pipe_tag=f'rag:{self.rag_pipe.id}' if self.rag_pipe else ''
        #if self.rag_pipe is not None:
        return f"{evaluator_tag}-{rag_pipe_tag}-callback_df.pkl"
    
    def on_epoch_end(self, args=None, state=None, control=None, **kwargs):

        if not self.evaluator.is_eval_epoch(state.epoch):
            self.experiment.logger.info(f"{self.name}\tepoch: {state.epoch}\tglobal step: {state.global_step} not an eval epoch")
            return #skip evaluation if not specified as eval epoch

        model=kwargs['model']
        model.eval()

        pipe_df=self.evaluator.eval_df.copy()
        if _isinstance(self.rag_pipe, RAGPipeline):
            self.rag_pipe.add_ranking_df(pipe_df)
            self.rag_pipe.add_briefing_df(pipe_df)
        pipe_df=self.evaluator.get_pipe_df(pipe_df, toc=self.toc, **kwargs) #add prompt messages, if rag
    
        callback_df=self.evaluator(
            llm_pipe=self.experiment.llm_pipe.__class__(model=model),
            pipe_df=pipe_df,
            **kwargs)

        self.save_df(callback_df, state)


class InjectionExperiment(FinetuningExperiment):

    def __int__(self, *args, rag_pipe: RAGPipeline = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.do_eval_base_model_rag=kwargs.get('do_eval_base_model_rag', False)

    def evaluate_base_model(self, **kwargs) -> None:
        self.logger.info(f"Evaluating base model")

        if hasattr(self, 'evaluators'):

            for evaluator in self.evaluators:
                evaluator(self.llm_pipe.get_model(), **kwargs)
    
    #def evaluate_rag(self, **kwargs) -> None:


