from beesup_llm import *
from beesup_llm.toolkit.setup_utils import *
from beesup_llm.toolkit.llm_utils import *

from beesup_llm.model import *
from beesup_llm.dataset import BaseDataset
from beesup_llm.training import *
from beesup_llm.extraction.extraction_pipeline import *


from transformers import TrainerCallback
class EvaluationCallback(TrainerCallback):
    def __init__(self, experiment):
        self.experiment = experiment
        pass

    def get_eval_df(self, model):
        model.eval()
        eval_df=self.experiment.eval_df
        eval_df=self.experiment.pipeline(eval_df, model_ref=model)
        return eval_df
    
    def save_eval_df(self, eval_df, global_step):
        eval_df.to_pickle(f"{self.experiment.path}/{str(global_step).zfill(4)}_eval_df.pkl")
        self.experiment.logger.info(f"Saved eval_df to {self.experiment.path}/{str(global_step).zfill(4)}_eval_df.pkl")

    def on_epoch_end(self, args, state, control, **kwargs):

        model=kwargs['model']
        eval_df=self.get_eval_df(model)
        self.save_eval_df(eval_df,state.global_step)
        

class ExtractionExperiment(BaseDirectory):
    type='extraction_experiment'

    @classmethod
    def get_pending(cls):

        parent_lab_path = extract_lab_path(os.getcwd())
        parent_dir_path = f'{parent_lab_path}/{cls.type}s'
        ids=get_ids(parent_dir_path)

        pending_list=[]
        for id in ids:
            config_dict = load_dict(f"{parent_dir_path}/{str(id).zfill(4)}_{cls.type}/config.yaml")
            if not config_dict['done']: 
                pending_list.append(config_dict)
        
        return pending_list
    


    def __init__(self, ref=None, dataset_ref=None, pipeline_ref=None, model_ref=None, trainer_ref=None, **kwargs):
        super().__init__(ref, **kwargs)

        self._config_key_order.extend([])
        self._config_keys_to_exclude.extend(['dataset_df','train_df','test_df','eval_df','train_ds','eval_ds','trainer','trainwrap','modelwrap','pipeline','dataset'])

        self._default_config=dict(
            done = False,
            do_eval_base_model=True,

            generation_config=dict(
                do_sample=False,
            ),

            dataloader_config=dict(
                batch_size=8,
            ),

            trainer_args=dict(
                seed=42,
                num_train_epochs=2,
                learning_rate=0.0002,
                output_dir=f"{self.path}",
                save_strategy='no',
                logging_strategy='steps',
                logging_steps=1,
                logging_first_step=True,
                
                eval_strategy='no',
                do_eval=False,
            ),

        )

        self.update_attributes(self._default_config, overwrite=False)



        model_ref = model_ref or getattr(self, 'model_config', None)
        if model_ref is not None:
            self.modelwrap=GenModelWrap.from_ref(model_ref)
            if not self.modelwrap.is_spawned(): self.modelwrap.spawn()
            self.model_config=self.get_updated_sub_config(self.modelwrap.get_config())

        dataset_ref = dataset_ref or getattr(self, 'dataset_config', None)
        if dataset_ref is not None:
            self.dataset=BaseDataset.from_ref(dataset_ref)
            if not self.dataset.is_spawned(): self.dataset.spawn()
            self.dataset_config=self.get_updated_sub_config(self.dataset.get_config())

        trainer_ref = trainer_ref or getattr(self, 'trainer_config', None)
        if trainer_ref is not None:
            self.trainwrap=BaseTrainerWrap.from_ref(trainer_ref)
            if not self.trainwrap.is_spawned(): self.trainwrap.spawn()
            self.trainer_config=self.get_updated_sub_config(self.trainwrap.get_config())

        pipeline_ref = pipeline_ref or getattr(self, 'pipeline_config', None)
        if pipeline_ref is not None:
            self.pipeline=ExtractionPipeline.from_ref(pipeline_ref)
            if not self.pipeline.is_spawned(): self.pipeline.spawn()
            self.pipeline_config=self.get_updated_sub_config(self.pipeline.get_config())

    def get_updated_sub_config(self, sub_config):

            ignore_keys=['type', 'id', 'name', 'path', 'parent_dir_path', 'parent_lab_path','timestamp_init']

            self_config=self.get_config()
            self_config={k:v for k,v in self_config.items() if k not in ignore_keys}

            sub_keys=sub_config.keys()
            self_config={k:v for k,v in self_config.items() if k  in sub_keys}

            for key in sub_keys:
                if key in ignore_keys: continue
                if hasattr(self, key): self.__delattr__(key)

            updated_sub_config=update_nested_dict(sub_config,self_config, overwrite=True)

            return updated_sub_config

    def load_data(self, **kwargs):
        self.logger.info(f"Loading Data")

        assert hasattr(self, 'dataset'), "Dataset must be assigned before loading data"
        assert hasattr(self, 'pipeline'), "Extraction Pipeline must be assigned before loading data"
        assert hasattr(self, 'modelwrap'), "ModelWrap must be assigned before loading data"

        train_df=self.dataset.get_df_splits('train')
        eval_df=self.dataset.get_df_splits(['test','eval'])

        self.modelwrap.load_training_tokenizer()
        self.modelwrap.load_inference_tokenizer()

        train_df=self.pipeline.prepare_df_for_finetuning(train_df)
        train_ds=self.pipeline.get_ds_for_finetuning(train_df, self.modelwrap.get_training_tokenizer()) # tokenizer type doesnt matter at this point, because padding is not applied

        eval_df=self.pipeline.prepare_df_for_completion(eval_df)
        eval_ds=self.pipeline.get_ds_for_completion(eval_df,  self.modelwrap.get_inference_tokenizer())

        self.train_df=train_df
        self.eval_df=eval_df

        self.train_ds=train_ds
        self.eval_ds=eval_ds

    def get_trainer(self,**kwargs):
        self.logger.info(f"Loading Trainer")
        
        self.load_data(**kwargs)

        model=kwargs.get('model')

        #trainer_args=self.get_updated_config(kwargs, 'trainer_args')
        #self.logger.info(f"Trainer Args: {trainer_args}")

    
        trainer= self.trainwrap.get_trainer(
            model=model,
            tokenizer=self.modelwrap.get_training_tokenizer(),
            train_dataset=self.train_ds,
            #args=trainer_args,
        )

        trainer.add_callback(EvaluationCallback(self))

        return trainer
    
    def evaluate_base_model(self,model):

        eval_callback=EvaluationCallback(self)
        
        eval_df=eval_callback.get_eval_df(model)
        eval_callback.save_eval_df(eval_df,0)


    def run(self,**kwargs):
        self.logger.info(f"RUNNING")
        self.timestam_run=get_datetime()
        base_model=self.modelwrap.get_model()

        if self.do_eval_base_model:
            self.evaluate_base_model(model=base_model)

        trainer=self.get_trainer(model=base_model,**kwargs)
        trainer_args=trainer.args.to_dict()
        save_yaml(trainer_args, f"{self.path}/trainer_args.yaml")
        
        trainer.train()

        self.done=True
        self.timestam_done=get_datetime()
        set_config(self.get_config())


    def get_evals_df(self):

        fns=os.listdir(self.path)
        fns=[fn for fn in fns if fn.endswith('eval_df.pkl')]
        evals_df=pd.DataFrame()
        for fn in fns:
            global_step=int(fn[:4])
            eval_df=pd.read_pickle(f"{self.path}/{fn}")
            eval_df['global_step']=global_step
            evals_df=pd.concat([evals_df,eval_df])

        evals_df.reset_index(drop=True, inplace=True)
        return evals_df




if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        print("Usage: python experiment_module.py <config_path>")
        sys.exit(1)
    
    config_path=sys.argv[1]
    print(f"Starting experiment from config: {config_path}")
    os.chdir(config_path)
    # Initialize and run the experiment
    experiment = ExtractionExperiment(config_path)
    experiment.run()

    