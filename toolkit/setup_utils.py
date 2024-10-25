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