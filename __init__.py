
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
            overview_row['id']=config_dict['id']
            overview_row['name']=config_dict['name']

            if 'path' in keypaths:
                overview_row['path']=f"{get_lab_path(config_dict['lab_name'])}/{config_dict['dir_name']}/{config_dict['name']}"
            #overview_row['path']=f"{get_lab_path(config_dict['lab_name'])}/{config_dict['dir_name']}/{config_dict['name']}"
            
            for keypath in keypaths:
                if keypath == 'path': continue

                logging.debug(f"id: {id}\tkeypath: {keypath}")

                value=None
                keypath_list=split_keypath(keypath)

                if keypath in config_dict.keys():

                    value=config_dict[keypath]

                elif len(keypath_list)==1:
                    ambiguous_keypaths = get_dict_keypaths(config_dict, keypath_list[0])

                    if len(ambiguous_keypaths)==1:
                        value=get_dict_value_from_keypath(config_dict, ambiguous_keypaths[0])

                    elif len(ambiguous_keypaths)>1:

                        ambiguous_values=[]
                        for ambiguous_keypath in ambiguous_keypaths:
                            ambiguous_values.append(get_dict_value_from_keypath(config_dict, ambiguous_keypath))
                        
                        if len(set(ambiguous_values))==1:
                            value=ambiguous_values[0]
                        else:
                            logging.warning(f"Multiple values found for '{keypath}' in config_dict: {ambiguous_values}")

                elif has_keypath(config_dict, keypath_list):
                    value=get_dict_value_from_keypath(config_dict, keypath_list)


                if value is not None:
                    for i in range(1,len(keypath_list)+1):
                        keypath_str='.'.join(keypath_list[len(keypath_list)-i:])
                        if keypath_str not in overview_row.keys():
                            break
                    
                    overview_row[keypath_str]=value

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
            #timestamp_init=get_timestamp(),
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
        self.logger.debug(f"config_dict: {config_dict}")
        self.update_config(config_dict, overwrite_if_conflict=True)


        #_paths only for internal use to allow use on different machines
        self._lab_path=get_lab_path(self.lab_name)
        self._dir_path=f"{self._lab_path}/{self.type}s"
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

        config_dict=self.get_config()

        if not has_key(config_dict,config_key,atomic_only=False):
            raise ValueError(f"config_key '{config_key}' not found in config_dict.")

        base_keypath=get_dict_keypath(config_dict,config_key)
        base_config=copy.deepcopy(get_dict_value_from_keypath(config_dict,base_keypath))
        self.logger.debug(f"base_config: {base_config}")

        if has_key(kwargs, config_key, atomic_only=False):
            mixin_keypath=get_dict_keypath(config_dict,config_key)
            mixin_config=copy.deepcopy(get_dict_value_from_keypath(config_dict,mixin_keypath))
            #reduced_kwargs=copy.deepcopy(del_dict_keypath(kwargs, mixin_keypath))
        
        else:
            mixin_config=filter_dict_keydict(kwargs, base_config)
            #reduced_kwargs=filter_dict_keydict(kwargs, mixin_config, invert=True)
        
        updated_config=update_dict(base_config, mixin_config, overwrite_if_conflict=True)
        self.logger.debug(f"updated_config: {updated_config}")
        return updated_config
    
    def is_spawned(self):
        return os.path.exists(f'{self._path}')

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


    def get_config(self):

        config=dict()
        for key in self._config_key_order:
            if key in config: continue
            if hasattr(self, key): 
                config[key]=getattr(self, key)

        further_items=self.__dict__
        further_items=filter_dict_valuetypes(further_items,valuetypes=[str,int,float,bool,dict,list,tuple,set,type(None)])
        further_items=filter_dict_keypatterns(further_items, [r'^_'], invert=True)
        further_items=filter_dict_keylist(further_items, self._config_keys_to_exclude, invert=True)

        config.update(further_items)

        return config

    def spawn(self):
        self.spawn_config()

    def __repr__(self):
        return f"{self.name} {type(self)}"






    

    

    






    
    
