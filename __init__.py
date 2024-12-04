
import os
import warnings
from .toolkit import *
from .toolkit.system import *
from .toolkit.dict_utils import *
from .toolkit.setup_utils import *



# from beesup_llm.dataset import BaseDataset
# from beesup_llm.training import BaseTraining
# from beesup_llm.inference import BaseTest
# from beesup_llm.model import BaseModelWrap
#from beesup_llm.evaluation import BaseEvaluation

import logging
from typing import Optional

import pandas as pd



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
    logger = logging.getLogger(__name__)

    @classmethod
    def get_dir_path(cls):

        lab_path=get_lab_path()
        dir_path=f"{lab_path}/{cls.type}s"

        return dir_path
    
    @classmethod
    def get_max_id(cls):
        dir_path=cls.get_dir_path()
        return get_max_id(dir_path)
    
    @classmethod
    def get_overview(cls, keypaths=[]):

        _keypaths=['path','id','name']+keypaths
        dir_path=cls.get_dir_path()

        if not os.path.exists(dir_path): raise FileNotFoundError(f"{dir_path} does not exist")

        overview_data=[]
        ids=get_ids(dir_path)

        for id in ids:
            overview_row=dict()

            config_path=f"{dir_path}/{str(id).zfill(4)}_{cls.type}/config.yaml"
            if not os.path.exists(config_path):
                cls.logger.warning(f"{config_path} does not exist")
                continue

            config_dict = load_dict(config_path)

            for keypath in _keypaths:

                #TODO: only set last key of keypath to overview dataframe
                keypath_list=split_keypath(keypath)
                value=get_value_from_keypath(config_dict, keypath_list)

                if value is not None: overview_row[keypath_list[-1]]=value
            
            overview_data.append(overview_row)
        
        return pd.DataFrame(overview_data)


    def __init__(self, ref=None, **kwargs):

        self._default_config=dict(
            type=self.type,
            id=None,
            name=None,
            dir_name=None,
            lab_name=None,
            rel_path=None,
            timestamp_init=get_timestamp(),
        )
        self._config_key_order=list(self._default_config.keys())
        self._config_keys_to_exclude=['logger','from_ref']

        self.logger.debug(f"ref: {ref}")
        self.logger.debug(f"kwargs: {kwargs}")

        init_kwargs=update_dict_smart(
            orign_dict=self._default_config, 
            mixin_dict=kwargs, 
            interpret_none_as_val=True,
            overwrite_if_conflict=True,
            allow_new_atomic_keys=False,
            allow_new_nested_keys=False,
            )
        
        self.logger.debug(f"init_kwargs: {init_kwargs}")
        config_dict=get_config_from_ref(ref,**init_kwargs)
        self.update_config(config_dict, overwrite_if_conflict=True)

        #_paths only for internal use to allow use on different machines
        self._lab_path=get_lab_path(self.lab_name)
        self._dir_path=self.get_dir_path()
        self._path=f"{self._dir_path}/{self.name}"

        self.logger.info(f"{self.name.upper()} initialised")

    def update_config(self, mixin_dict, overwrite_if_conflict=True, interpret_none_as_val=True):
        
        #self.logger.debug(f"mixin_dict: {mixin_dict}")

        updated_config_dict = update_dict(
            self.get_config(),
            mixin_dict,
            overwrite_if_conflict=overwrite_if_conflict,
            interpret_none_as_val=interpret_none_as_val
            )
        
        #self.logger.debug(f"updated_config_dict: {mixin_dict}")
        for k, v in updated_config_dict.items(): setattr(self, k, v)
        return
    
    def update_config_smart(self, 
                            mixin_dict, 
                            interpret_none_as_val=True, 
                            overwrite_if_conflict=True, 
                            allow_new_atomic_keys=False, 
                            allow_new_nested_keys=False
                            ):
        
        #self.logger.debug(f"mixin_dict: {mixin_dict}")
        
        updated_config_dict=update_dict_smart(
            self.get_config(),
            mixin_dict,
            interpret_none_as_val=interpret_none_as_val,
            overwrite_if_conflict=overwrite_if_conflict,
            allow_new_atomic_keys=allow_new_atomic_keys,
            allow_new_nested_keys=allow_new_nested_keys
            )
        
        #self.logger.debug(f"updated_config_dict: {mixin_dict}")
        for k, v in updated_config_dict.items(): setattr(self, k, v)
        return
        
   
    def reinit_config(self):
        new_config=get_config_from_type(type=self.type)
        self.update_config(new_config, overwrite_if_conflict=True)

    def get_updated_config(self, kwargs, config_key='some_config'):

        base_config=copy.deepcopy(getattr(self,config_key))

        self.logger.debug(f"base_config: {base_config}")

        if config_key in kwargs: kwargs.update(kwargs.get(config_key))

        self.logger.debug(f"kwargs: {kwargs}")

        kwargs={k:v for k,v in kwargs.items() if k in base_config.keys()}

        base_config.update(kwargs)

        self.logger.debug(f"updated_config: {base_config}")
        return base_config
    
    def spawn_config(self):

        if not os.path.exists(f'{self._dir_path}'):
            os.makedirs(f'{self._dir_path}', exist_ok=False)

        
        if os.path.exists(f'{self._path}/config.yaml'):
            old_config = load_dict(f'{self._path}/config.yaml')

            if old_config != self.get_config():
                warnings.warn(f"Already existing. Spawn as new instance.")
                self.reinit_config()


        if not os.path.exists(f'{self._path}'):
            os.makedirs(f'{self._path}', exist_ok=False)

        set_config(self.get_config(), path=self._path)
        self.logger.info(f"{self.name.upper()} config spawned at {self.rel_path}")

    def update_config_smart(self, mixin_dict, interpret_none_as_val=True, overwrite_if_conflict=True, allow_new_atomic_keys=False, allow_new_nested_keys=False):

        updated_config_dict=update_dict_smart(
            self.get_config(),
            mixin_dict,
            overwrite_if_conflict=overwrite_if_conflict,
            interpret_none_as_val=interpret_none_as_val,
            allow_new_atomic_keys=allow_new_atomic_keys,
            allow_new_nested_keys=allow_new_nested_keys
            )
        
        for k, v in updated_config_dict.items(): setattr(self, k, v)
        return

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






    

    

    






    
    
