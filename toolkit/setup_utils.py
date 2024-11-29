import re
import os
import yaml
import copy

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

def get_ids(parent_path=None, type='', from_id=0, to_id=10e9):
    id_pattern=r'^(\d{4,5})_'+type

    if not os.path.exists(f'{parent_path}'): return []

    fns=os.listdir(parent_path)
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

def set_config(config):
    save_yaml(config, f'{config["path"]}/config.yaml')

def extract_lab_path(path):

    # Traverse upwards to find the first directory that ends with '_lab'
    while path != os.path.dirname(path):  # Loop until root directory is reached
        dir_name = os.path.basename(path)
        if dir_name.endswith('_lab'):
            return path # Stop once the first matching directory is found

        # Move up one directory level
        path = os.path.dirname(path)

    # If no directory ending with '_lab' was found
    raise FileNotFoundError("No directory ending with '_lab' found in the path.")

def extract_type_from_path(path):
    match = re.search(r"/(\w+)s/\d+_\1\b", path)
    if match:
        return match.group(1)
    else:
        return None

def extract_type(input_obj):

    the_type=None

    if isinstance(input_obj, str):
        the_type=extract_type_from_path(input_obj)

    elif hasattr_or_key(input_obj,'type'):
        the_type=getattr_or_key(input_obj,'type')


    return the_type

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
    if getattr_or_key(config, 'path') is None: return False
    if getattr_or_key(config, 'name') is None: return False
    if getattr_or_key(config, 'timestamp_init') is None: return False

    return True

def get_config_from_id(id, type=None):

    parent_lab_path = extract_lab_path(os.getcwd())
    parent_dir_path = f'{parent_lab_path}/{type}s'  # derive the parent directory path from the type (e.g. dataset -> datasets)
    assert os.path.exists(f'{parent_dir_path}'), f"parent directory {parent_dir_path} does not exist"
    config_dict = load_dict(f"{parent_dir_path}/{str(id).zfill(4)}_{type}/config.yaml")
    return config_dict

def get_config_from_path(path):
    if not path.endswith('config.yaml'):
        path = f"{path}/config.yaml"
    assert path.endswith('config.yaml'), "path must point to a 'config.yaml'"
    config_dict = load_dict(path)
    return config_dict

def get_config_from_dict(the_dict):

    if not is_valid_config(the_dict):

        if the_dict.get('path', None): return get_config_from_path(the_dict['path'])
        if the_dict.get('id', None) and the_dict.get('type',None): return get_config_from_id(the_dict['id'], the_dict['type'])
        if the_dict.get('type', None): return get_config_from_type(the_dict['type'])
        else : 
            logging.info(f"Invalid config dictionary: {the_dict}")
            raise ValueError("Invalid config dictionary.")

    return the_dict

def get_config_from_obj(obj):
    assert hasattr(obj, 'get_config'), "object must have a 'get_config' method"
    return obj.get_config()

def get_config_from_type(type=None):

    parent_lab_path = extract_lab_path(os.getcwd())
    parent_dir_path = f'{parent_lab_path}/{type}s'  # derive the parent directory path from the type (e.g. dataset -> datasets)

    # if not os.path.exists(f'{parent_dir_path}'):
    #     os.makedirs(f'{parent_dir_path}', exist_ok=False)

    id = get_max_id(parent_dir_path) + 1
    name = f"{str(id).zfill(4)}_{type}"

    return dict(
        type=type,
        id=id,
        name=name,
        path=f"{parent_dir_path}/{name}",
        timestamp_init=get_timestamp(),
        parent_lab_path=parent_lab_path,
        parent_dir_path=parent_dir_path
    )

def get_config_from_model(model,**kwargs):

    config_dict=get_config_from_type(**kwargs)
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
        return get_config_from_id(ref, type=kwargs['type'])
    
    elif isinstance(ref, str):
        return get_config_from_path(ref)
    
    elif isinstance(ref, dict):
        ref.update(kwargs)
        return get_config_from_dict(ref)
    
    elif hasattr(ref, 'get_config'):
        return get_config_from_obj(ref)
    
    else:
        import torch
        if isinstance(ref, torch.nn.Module):
            return get_config_from_model(ref,**kwargs)
