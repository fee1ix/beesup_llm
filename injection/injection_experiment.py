import copy
import pandas as pd

from beesup_llm import _isinstance
from beesup_llm.finetuning_experiment import *
from beesup_llm.injection.rag import RAGPipeline
from beesup_llm.injection.injection_evaluation import *

class InjectionCallback(EvaluatorCallback):

    def __init__(self, *args, rag_pipe: RAGPipeline=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.toc = None
        if hasattr(self.experiment,'taxomizer'):
            self.toc=self.experiment.taxomizer.toc
        self.experiment.logger.info(f"{self.toc=}")

        if _isinstance(rag_pipe, RAGPipeline):
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

        pipe_df=self.evaluator.eval_df.iloc[:4].copy()

        if hasattr(self,'rag_pipe'):
            self.rag_pipe.add_ranking_df(pipe_df)
            self.rag_pipe.add_briefing_df(pipe_df)

        pipe_df=self.evaluator.get_pipe_df(pipe_df, toc=self.toc, **kwargs) #add prompt messages, if rag
    
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

    def __init__(self, *args, rag_pipe: RAGPipeline = None, **kwargs):

        self.do_eval_base_model_rag=kwargs.get('do_eval_base_model_rag', False)
        self.do_eval_lora_model_rag=kwargs.get('do_eval_lora_model_rag', False)

        super().__init__(*args, **kwargs)

        if hasattr(self, 'labh'):
            rag_pipe=self.labh.handle_object(locals(),'rag_pipe')

        if _isinstance(rag_pipe, RAGPipeline):
            self.rag_pipe=rag_pipe
            self.rag_pipe.llm_pipe=getattr(self, 'llm_pipe', None)
            

    def evaluate_base_model(self, rag_pipe:RAGPipeline=None, **kwargs) -> None:
        super().evaluate_base_model(callback_class=InjectionEvaluator, rag_pipe=rag_pipe, **kwargs)
        return
    
    def run(self, **kwargs) -> None:

        self.timestamp_start=get_timestamp()
        self.llm_pipe.prepare_inference()
        self.load_data(**kwargs)



        if self.do_eval_base_model_rag:
            self.evaluate_base_model(rag_pipe=self.rag_pipe, **kwargs)
        
        if self.do_eval_base_model:
            self.evaluate_base_model(**kwargs)
        

        if (self.do_eval_lora_model or self.do_train):

            self.train_ds=self.train_ds.select([0, 1, 2])
            self.load_trainer(**kwargs)

        if self.do_eval_lora_model_rag:
            for evaluator in self.evaluators:
                evaluator_callback=InjectionCallback(evaluator, self, rag_pipe=self.rag_pipe, **kwargs)
                self.trainer.add_callback(evaluator_callback)

        if self.do_eval_lora_model:
            for evaluator in self.evaluators:
                evaluator_callback=InjectionCallback(evaluator, self, **kwargs)
                self.trainer.add_callback(evaluator_callback)


        if self.do_train:
            self.train(**kwargs)

        self.done=True
        self.timestamp_done=get_timestamp()
        return
        



        






