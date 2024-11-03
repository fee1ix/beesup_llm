import re
import os
import yaml

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
    fns=os.listdir(parent_path)
    fns=[fn for fn in fns if bool(re.match(id_pattern,fn))]
    ids=[int(re.findall(id_pattern,fn)[0]) for fn in fns]
    ids=[id for id in ids if (id >=from_id) and (id <=to_id)]

    return ids

def set_config(config):
    with open(f'{config["path"]}/config.yaml', 'w') as file:
        yaml.dump(config, file, sort_keys=False, default_flow_style=False)

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


def hasattr_or_key(obj, key):
    # Check if it's a dictionary and contains the key
    if isinstance(obj, dict):
        return key in obj
    # Otherwise, check if it's an object and has the attribute
    return hasattr(obj, key)

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

# def gather_attributes(input_obj,**kwargs):

#     attributes_dict=filter_attributes(input_obj, **kwargs)

#     if hasattr(input_obj,'__class__'):

#         for cls in input_obj.__class__.__mro__:

#             if not hasattr(cls, '__dict__'): continue

#             attributes_dict.update(filter_attributes(vars(cls), **kwargs))

