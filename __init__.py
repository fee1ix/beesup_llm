
import os
import warnings
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
    Base class for all directories in the lab.
    """

    type = 'directory'
    _config_key_order=['type', 'id', 'name', 'path', 'parent_dir_path', 'parent_lab_path']
    _config_keys_to_exclude=['logger','from_ref']
    logger = logging.getLogger(__name__)

    def __init__(self, ref=None, **kwargs):

        kwargs.update(self.get_config())
        self.logger.debug(f"{self.__class__} ref={ref}, kwargs = {kwargs}\n")

        for k,v in kwargs.items():
            setattr(self,k,v)

        config_dict=get_config_from_ref(ref,**kwargs)

        self.update_attributes(config_dict, overwrite=True)
        self.logger.info(f"{self.name.upper()} initialised")

    def reinit_config(self):
        new_config=get_config_from_none(type=self.type)
        self.update_attributes(new_config, overwrite=True)

    def get_updated_config(self, kwargs, config_key='some_config'):

        base_config=copy.deepcopy(getattr(self,config_key))

        if config_key in kwargs: kwargs.update(kwargs.get(config_key))

        kwargs={k:v for k,v in kwargs.items() if k in base_config.keys()}

        base_config.update(kwargs)
        return base_config

    def spawn_config(self):

        if not os.path.exists(f'{self.parent_dir_path}'):
            os.makedirs(f'{self.parent_dir_path}', exist_ok=False)

        
        if os.path.exists(f'{self.path}/config.yaml'):
            old_config = load_dict(f'{self.path}/config.yaml')

            if old_config != self.get_config():
                warnings.warn(f"Already existing. Spawn as new instance.")
                self.reinit_config()


        if not os.path.exists(f'{self.path}'):
            os.makedirs(f'{self.path}', exist_ok=False)

        set_config(self.get_config())
        self.logger.info(f"{self.name.upper()} config spawned at {self.path}")


    def update_attributes(self, new_dict, overwrite=True):
        updated_config = update_nested_dict(self.get_config(), new_dict, overwrite)

        for k, v in updated_config.items():
            setattr(self, k, v)
        
        return

    def get_max_id(self):
        assert hasattr(self, 'parent_dir_path'), "parent_dir_path must be defined"
        return get_max_id(self.parent_dir_path)
    
    def is_spawned(self):
        return os.path.exists(f'{self.path}')

    def get_config(self):

        config = {k: getattr(self, k) for k in self._config_key_order if hasattr(self, k)}
        config.update({k: v for k, v in self.__dict__.items() if 
                       (k not in self._config_keys_to_exclude) and
                       (not k.startswith('_')) and
                       #(not isinstance(v, (pd.DataFrame))) and
                       #(isinstance(k,(str,int,float,dict,list,tuple))) and
                       (k not in self._config_key_order)})

        return config

    def spawn(self):
        self.spawn_config()

    def __repr__(self):
        return f"{self.name} {type(self)}"






    

    

    






    
    
