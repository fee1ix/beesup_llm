
import os
from .toolkit import *
from .toolkit.setup_utils import *

# from beesup_llm.dataset import BaseDataset
# from beesup_llm.training import BaseTraining
# from beesup_llm.inference import BaseTest
# from beesup_llm.model import BaseModelWrap
#from beesup_llm.evaluation import BaseEvaluation

import logging
from typing import Optional

import pytz
import datetime
TIMEZONE = pytz.timezone('Europe/Berlin')

class Lab(object):

    def __init__(self, path=None):

        if path is None:
            self.deduce_from_cwd()

    
    def deduce_from_cwd(self):

        # Start from the current working directory
        current_path = os.getcwd()

        # Traverse upwards to find the first directory that ends with '_lab'
        while current_path != os.path.dirname(current_path):  # Loop until root directory is reached
            dir_name = os.path.basename(current_path)
            if dir_name.endswith('_lab'):
                self.name = dir_name
                self.path = current_path
                return  # Stop once the first matching directory is found

            # Move up one directory level
            current_path = os.path.dirname(current_path)

        # If no directory ending with '_lab' was found
        raise FileNotFoundError("No directory ending with '_lab' found in the current or parent directories.")

    
    def __repr__(self):

        total_size = 0
        file_count = 0
        dir_count = 0
        
        for root, dirs, files in os.walk(self.path):
            # Increment the directory count
            dir_count += len(dirs)
            
            # Increment file count and total size
            for file in files:
                file_count += 1
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
        
        # Convert total size from bytes to a more readable format (MB)
        total_size_mb = total_size / (1024 * 1024)

        return f"""
Lab-Path: {self.path}
Total number of files: {file_count}
Total number of subdirectories: {dir_count}
Total size of files: {total_size_mb:.2f} MB
        """.strip()

class BaseDirectory(object):
    """
    Base class for subdirectories inside a lab
    """
    def __new__(cls, ref=None, skip_new=False, **kwargs):

        if skip_new:
            print('skip_new')
            return super().__new__(cls)

        if is_valid_config(ref):
            print('valid config')
            config = ref

        else: 
            print('not valid config')
            if 'type' not in kwargs: kwargs['type']='directory'
            config=get_config_from_ref(ref, **kwargs)

        if (cls is BaseDirectory):

            if config['type'] == 'dataset':
                from beesup_llm.dataset import BaseDataset
                return BaseDataset(config)
            
            elif config['type'] == 'training': 
                from beesup_llm.training import BaseTraining
                return BaseTraining(config)

            elif config['type'] == 'test': 
                from beesup_llm.inference import BaseTest
                return BaseTest(config)
            
            elif config['type'] == 'model': 
                from beesup_llm.model import BaseModelWrap
                return BaseModelWrap(config)
        
        return super().__new__(cls)

    
    def __init__(self, ref=None, **kwargs):
        self._config_key_order=['type', 'id', 'name', 'path', 'parent_dir_path', 'parent_lab_path']
        self._config_keys_to_exclude=['logger']

        if is_valid_config(ref): self.update_attributes(ref,overwrite=True)

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"{self.name.upper()} initialised")

    def legacy__init__(self, ref=None):

        self._config_key_order=['type', 'id', 'name', 'path', 'parent_dir_path', 'parent_lab_path']
        self._config_keys_to_exclude=['logger']

        if getattr(self, '_ref', None) is not None:
            ref=self._ref; del self._ref
        
        if not hasattr(self, 'type'): self.type = 'directory'

        if isinstance(ref, int): self.__init__from_id(ref)

        elif isinstance(ref, str): self.__init__from_path(ref)

        elif isinstance(ref, dict): self.__init__from_config(ref)

        elif hasattr(ref, 'get_config'): self.__init__from_config(ref.get_config())

        else: #initialize a new instance

            self.parent_lab_path=extract_lab_path(os.getcwd())
            self.parent_dir_path=f'{self.parent_lab_path}/{self.type}s' # derive the parent directory path from the type (e.g. dataset -> datasets)

            if not os.path.exists(f'{self.parent_dir_path}'):
                os.makedirs(f'{self.parent_dir_path}', exist_ok=False)

            self.id=self.get_max_id()+1
            self.name=f"{str(self.id).zfill(4)}_{self.type}"
            self.path=f"{self.parent_dir_path}/{self.name}"

            self.datetime_init=datetime.datetime.now(TIMEZONE)


        self.logger = logging.getLogger(__name__)
        self.logger.info(f"{self.name.upper()} initialised")

    def __init__from_config(self,config):
        for k, v in config.items(): 
            if (v is not None) and (not hasattr(self, k)):
                setattr(self, k, v)

    def __init__from_id(self,id):
        assert hasattr(self, 'type'), "missing type key"

        self.parent_lab_path=extract_lab_path(os.getcwd())
        self.parent_dir_path=f'{self.parent_lab_path}/{self.type}s' # derive the parent directory path from the type (e.g. dataset -> datasets)
        
        assert os.path.exists(f'{self.parent_dir_path}'), f"parent directory {self.parent_dir_path} does not exist"

        config = load_dict(f"{self.parent_dir_path}/{str(id).zfill(4)}_{self.type}/config.yaml")
        self.__init__from_config(config)

    def __init__from_path(self,path):

        if not path.endswith('config.yaml'): 
            path=f"{path}/config.yaml"

        assert path.endswith('config.yaml'), "path must point to a 'config.yaml'"
        config=load_dict(path)

        self.__init__from_config(config)

    def update_attributes(self, new_dict, overwrite=True):
        updated_config = update_nested_dict(self.get_config(), new_dict, overwrite)

        for k, v in updated_config.items():
            setattr(self, k, v)
        
        return


    # def get_max_id(self):
    #     assert hasattr(self, 'parent_dir_path'), "parent_dir_path must be defined"

    #     max_id=0

    #     id_list = get_ids(self.parent_dir_path)
    #     if id_list: max_id = max(id_list)

    #     return max_id

    def get_config(self):

        config = {k: getattr(self, k) for k in self._config_key_order if hasattr(self, k)}
        config.update({k: v for k, v in self.__dict__.items() if 
                       (k not in self._config_keys_to_exclude) and
                       (not k.startswith('_')) and
                       #(isinstance(k,(str,int,float,dict,list,tuple))) and
                       (k not in self._config_key_order)})

        return config

    def spawn(self):

        if not os.path.exists(f'{self.path}'):
            os.makedirs(f'{self.path}', exist_ok=False)

        set_config(self.get_config())
        logging.info(f"{self.name.upper()} spawned at {self.path}")

    def get_datetime(self):
        return datetime.datetime.now(TIMEZONE)

    def __repr__(self):

        return f"{self.name} {type(self)}"






    

    

    






    
    
