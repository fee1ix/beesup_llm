import re
import os
import yaml

def load_yaml(path):

    with open(path, 'r') as file:
        the_yaml = yaml.safe_load(file)
    
    return the_yaml

def load_dict(input):

    if isinstance(input, dict): return input

    elif isinstance(input, str) and input.endswith('.yaml'): return load_yaml(input)

    else: return None
        #raise ValueError("Input must be a dictionary or a string path.")


def get_ids(parent_path=None, type='', from_id=0, to_id=10e9):
    id_pattern=r'^(\d{4,5})_'+type
    fns=os.listdir(parent_path)
    fns=[fn for fn in fns if bool(re.match(id_pattern,fn))]
    ids=[int(re.findall(id_pattern,fn)[0]) for fn in fns]
    ids=[id for id in ids if (id >=from_id) and (id <=to_id)]

    return ids