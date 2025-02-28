import re
import os
import yaml
import copy
import logging

from .system import *
from .dict_utils import *


def load_yaml(path):

    with open(path, 'r') as file:
        the_yaml = yaml.safe_load(file)
    
    return the_yaml

def load_dict(ref):

    if isinstance(ref, dict): return ref

    elif isinstance(ref, str) and ref.endswith('.yaml'): return load_yaml(ref)

    else: return None
        #raise ValueError("Input must be a dictionary or a string path.")

def get_ids(path=None, type='', from_id=0, to_id=10e9):
    id_pattern=r'^(\d{4,5})_'+type

    if not os.path.exists(f'{path}'): return []

    fns=os.listdir(path)
    fns=[fn for fn in fns if bool(re.match(id_pattern,fn))]
    ids=[int(re.findall(id_pattern,fn)[0]) for fn in fns]
    ids=[id for id in ids if (id >=from_id) and (id <=to_id)]

    return ids

def get_max_id(path):
    max_id = 0
    id_list = get_ids(path)
    if id_list: max_id = max(id_list)
    return max_id

def save_yaml(the_yaml, path):

    with open(path, 'w') as file:
        yaml.dump(the_yaml, file, sort_keys=False, default_flow_style=False)

def set_config(config,path=None):

    if not path:
        lab_name=config['lab_name']
        abs_path = get_path_until(lab_name,include_dir=False)
        path = f"{abs_path}/{config['rel_path']}"
    
    elif not path.endswith('config.yaml'):
        path = f"{path}/config.yaml"

    save_yaml(config, path)


from pathlib import Path
def get_lab_name(path=None):
    if not path: path = os.getcwd()

    lab_names=[p for p in list(Path(path).parts) if p.endswith('_lab')]
    if len(lab_names)==0: raise ValueError('No lab name found in path')
    if len(lab_names)>1: warnings.warn('Multiple lab names found in path. Using last one.')
    return lab_names[-1]

def get_path_until(dir_name, path=None, include_dir=False):
    if not path: path = os.getcwd()

    path = Path(path)
    parts = []
    for part in path.parts:
        if part == dir_name:
            if include_dir: parts.append(part)
            break
        parts.append(part)
    
    if dir_name not in path.parts:
        raise ValueError(f"Directory '{dir_name}' not found in the path.")
    
    return str(Path(*parts))

def get_lab_path(lab_name=None):

    current_lab_name = get_lab_name()
    current_lab_path = get_path_until(current_lab_name, include_dir=True)

    if (lab_name is None) or (lab_name == current_lab_name):
        return current_lab_path
    
    else: #now that we know the current lab path, we can search one level up for the target lab
        labs_path = os.path.dirname(current_lab_path)
        lab_names = [fn for fn in os.listdir(labs_path) if fn.endswith('_lab')]

        if lab_name not in lab_names:
            raise ValueError(f"Lab '{lab_name}' not found in '{labs_path}'")
        
        return f"{labs_path}/{lab_name}"



def update_nested_dict(original, updates, overwrite=True):
    for key, value in updates.items():
        # Check if both original[key] and value are dictionaries
        if key in original and isinstance(original[key], dict) and isinstance(value, dict):
            # Recurse if both are dictionaries
            original[key] = update_nested_dict(original[key], value, overwrite)

        elif key not in original or original[key] is None:
            original[key] = value

        elif overwrite:
            original[key] = value

    return original

def setattr_or_key(obj, key, val):
    #global obj

    if isinstance(obj, dict):
        obj[key] = val
    
    else:
        setattr(obj,key,val)

def hasattr_or_key(obj, key):
    # Check if it's a dictionary and contains the key
    if isinstance(obj, dict):
        return key in obj
    # Otherwise, check if it's an object and has the attribute
    return hasattr(obj, key)

def hasattrs_or_keys(obj, keys):
    return all(hasattr_or_key(obj, key) for key in keys)

def getattr_or_key(obj, key, val=None):
    # Check if it's a dictionary and contains the key

    if isinstance(obj, dict):
        if key in obj: val = obj[key]
    
    val = getattr(obj, key, val)

    return val


def filter_attributes(
        input_obj,
        exclude_prefixes=['_','__'],
        include_prefixes=[],
        exclude_types=[],
        include_types=[str,int,float,bool],
        ):
    
    if isinstance(input_obj, dict):
        input_dict=input_obj
        

    else:
        if hasattr(input_obj,'__dict__'):
            input_dict = input_obj.__dict__
    
    del input_obj
    filtered_dict = {}
    
    for key, value in input_dict.items():
        # Check if the attribute starts with any of the exclude prefixes
        if any(key.startswith(prefix) for prefix in exclude_prefixes):
            continue  # Skip this attribute

        # If include_prefixes is specified, check if the attribute starts with any of them
        if include_prefixes and not any(key.startswith(prefix) for prefix in include_prefixes):
            continue  # Skip this attribute if it doesn't match any include_prefix

        # Check if the value's type is in the exclude_types list
        if any(isinstance(value, t) for t in exclude_types):
            continue  # Skip this attribute

        # If include_types is specified, check if the value's type matches any of them
        if include_types and not any(isinstance(value, t) for t in include_types):
            continue  # Skip if it doesn't match any include_type

        # Add the attribute to the result if it passed all filters
        filtered_dict[key] = value

    return filtered_dict

def filter_kwargs(the_kwargs, allowed_keys=[], ref=None, exclude_prefixes=['_']):
  
    if hasattr(ref,'__dict__'):
        allowed_keys+=[k for k in ref.__dict__.keys() if not any([k.startswith(prefix) for prefix in exclude_prefixes])]
    
    if hasattr(ref,'__annotations__'):
        allowed_keys+=[k for k in ref.__annotations__.keys() if not any([k.startswith(prefix) for prefix in exclude_prefixes])]
    
    if isinstance(ref,dict):
        allowed_keys+=[k for k in ref.keys() if not any([k.startswith(prefix) for prefix in exclude_prefixes])]
    
    if isinstance(ref,list):
        allowed_keys+=[k for k in ref if not any([k.startswith(prefix) for prefix in exclude_prefixes])]

    filtered_kwargs={k:v for k,v in the_kwargs.items() if k in allowed_keys}
 
    return filtered_kwargs



def get_cls_attrs(cls):
    return {key: value for key, value in cls.__dict__.items() if not key.startswith("__") and not callable(value) and not isinstance(value, classmethod)}

def split_keypath(keypath):
    if isinstance(keypath,str):
        keypath_list=re.split(r'[\./]',keypath)
    elif isinstance(keypath,list):
        keypath_list=keypath
    return keypath_list

def get_value_from_keypath(the_dict, keypath):

    keypath_list=split_keypath(keypath)

    def _recursive(the_dict, keypath_list):
        key = keypath_list[0]
        if key in the_dict:
            if len(keypath_list)>1:
                return _recursive(the_dict[key], keypath_list[1:])
            else:
                return the_dict[key]
        else:
            return None
        
    return _recursive(the_dict, keypath_list)

def is_valid_config(config):
    if getattr_or_key(config, 'type') is None: return False
    if getattr_or_key(config, 'id') is None: return False
    if getattr_or_key(config, 'name') is None: return False
    if getattr_or_key(config, 'dir_name') is None: return False
    if getattr_or_key(config, 'lab_name') is None: return False
    #if getattr_or_key(config, 'timestamp_init') is None: return False

    return True

def get_config_from_id(id, **kwargs):

    type=kwargs.get('type',None)
    lab_name=kwargs.get('lab_name',None)

    if not lab_name:
        lab_name = get_lab_name(os.getcwd())
    
    logging.debug(f"type: {type}, lab_name: {lab_name}")

    dir_name = f'{type}s'  # derive the parent directory path from the type (e.g. dataset -> datasets)

    lab_path = get_lab_path(lab_name)
    dir_path = f"{lab_path}/{dir_name}"

    assert os.path.exists(dir_path), f"{dir_path} does not exist"
    config_dict = load_dict(f"{dir_path}/{str(id).zfill(4)}_{type}/config.yaml")

    return config_dict

def get_config_from_path(path):
    if not path.endswith('config.yaml'):
        path = f"{path}/config.yaml"
    assert path.endswith('config.yaml'), "path must point to a 'config.yaml'"
    config_dict = load_dict(path)
    return config_dict

def get_config_from_dict(the_dict):

    logging.debug(f"the_dict: {the_dict}")

    if not is_valid_config(the_dict):

        if the_dict.get('name', None) is not None:
            the_dict['id'] = int(the_dict['name'][:4])
            #the_dict['type'] = the_dict['name'][5:]

        if all([bool(the_dict.get(k, None)) for k in ['type','id']]):
            return get_config_from_id(**the_dict)
        
        if all([bool(the_dict.get(k, None)) for k in ['type']]):
            return get_config_from_type(the_dict['type'])
        

        # if all([k in the_dict for k in ['type','id']]):
        #     return get_config_from_id(**the_dict)

        #if all([k in the_dict for k in ['type','name']]):


        # if the_dict.get('type', None): return get_config_from_path(the_dict['path'])
        # if the_dict.get('id', None) and the_dict.get('type',None): return get_config_from_id(the_dict['id'], the_dict['type'])
        # if the_dict.get('type', None): return get_config_from_type(the_dict['type'])
        else : 
            return the_dict
            # logging.info(f"Invalid config dictionary: {the_dict}")
            # raise ValueError("Invalid config dictionary.")

    return the_dict

def get_config_from_obj(obj):
    assert hasattr(obj, 'get_config'), "object must have a 'get_config' method"
    return obj.get_config()

def get_config_from_type(type=None):

    lab_name = get_lab_name(os.getcwd())
    dir_name = f'{type}s'  # derive the parent directory path from the type (e.g. dataset -> datasets)
    dir_path = f"{get_path_until(lab_name, include_dir=False)}/{lab_name}/{dir_name}"

    id = get_max_id(dir_path) + 1

    name = f"{str(id).zfill(4)}_{type}"
    rel_path=f"{lab_name}/{dir_name}/{name}"

    return dict(
        type=type,
        id=id,
        name=name,
        rel_path=rel_path,
        lab_name=lab_name,
        dir_name=dir_name,
        timestamp_init=get_timestamp(),
    )

def get_config_from_model(model,**kwargs):

    config_dict=get_config_from_type(kwargs.get('type'))
    config_dict.update(
        name_or_path=getattr_or_key(model, 'name_or_path'),
        model=model
    )

    return config_dict

import logging

def get_config_from_ref(ref, **kwargs):

    logging.debug(f"get_config_from_ref: {ref}, kwargs: {kwargs}\n")

    if ref is None:
        return get_config_from_type(type=kwargs['type'])
    
    if isinstance(ref, int):
        kwargs.pop('id',None)
        logging.debug(f"kwargs: {kwargs}")
        return get_config_from_id(ref, **kwargs)
    
    elif isinstance(ref, str):
        return get_config_from_path(ref)
    
    elif isinstance(ref, dict):
        logging.debug(f"kwargs: {kwargs}")
        ref=update_dict(ref, kwargs, overwrite_if_conflict=False)
        #ref.update(kwargs)
        return get_config_from_dict(ref)
    
    elif hasattr(ref, 'get_config'):
        return get_config_from_obj(ref)
    
    else:
        import torch
        if isinstance(ref, torch.nn.Module):
            return get_config_from_model(ref,**kwargs)
