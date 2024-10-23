
import os
from .toolkit import *
from .toolkit.setup_utils import *

base_directory_config=dict(
    type = 'directory',
    sub_directories = []
)

class BaseDirectory(object):

    def __init__(self, config=base_directory_config, parent_object=None):

        config=load_dict(config)

        if isinstance(parent_object, type(None)):
            self.parent_path, self.name = os.path.split(os.getcwd())

        elif parent_object.type == 'directory':
            self.parent_path=parent_object.path
        
        elif parent_object.type == 'lab':
            self.parent_path=f"{parent_object.path}/{config['type']}s"

        for k, v in config.items(): setattr(self, k, v)
        
        self.get_id()
    
        if not hasattr(self, 'name'):
            self.name=f"{str(self.id).zfill(4)}_{self.type}"

        self.path='{parent_path}/{name}'.format(**self.__dict__)
        self.config=config

    def get_id(self):

        id_list = get_ids(self.parent_path)

        if id_list:
            max_id = max(id_list)
            self.id = max_id + 1

        else:
            self.id = 1

        return self.id

    def build(self):

        if not os.path.exists(f'{self.path}'):
            os.makedirs(f'{self.path}', exist_ok=False)

        for sub_dir in getattr(self,'sub_directories', []):
            if not os.path.exists(f'{self.path}/{sub_dir}'):
                os.makedirs(f'{self.path}/{sub_dir}', exist_ok=False)

    def __str__(self):
        return f"{self.path}"
  

base_lab_config=dict(
    type = 'lab',
    sub_directories = ['datasets','trainings','tests','evaluations']
)

class BaseLab(BaseDirectory):

    def __init__(self,config=base_lab_config):
        super().__init__(config)





    

    

    






    
    
