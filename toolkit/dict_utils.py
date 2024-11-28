import re
import copy
import logging

def is_nested(a_dict):
    return any(isinstance(v, dict) for v in a_dict.values())

def is_unique(element, a_list):
    return a_list.count(element) == 1

def has_duplicates(a_list):
    return len(a_list) != len(set(a_list))

def get_duplicates(a_list):
    return list(set([x for x in a_list if a_list.count(x) > 1]))

def get_keys(the_dict, atomic_only=False):
    keys = []
    for k, v in the_dict.items():
        if not atomic_only or not isinstance(v, dict):
            keys.append(k)
        if isinstance(v, dict):
            keys.extend(get_keys(v, atomic_only))
    return keys


def filter_dict_valuetypes(the_dict, valuetypes=[], invert=False):

    if (dict in valuetypes) and not invert: valuetypes.remove(dict)
    if (dict not in valuetypes) and invert: valuetypes.append(dict)

    def _recursive(the_dict):

        new_dict=dict()

        for k,v in the_dict.items():

            if isinstance(v, tuple(valuetypes)) ^ invert:
                new_dict[k]=v
                
            elif isinstance(v, dict) and not invert:
                new_dict[k] = _recursive(v)
        
        return new_dict
    
    return _recursive(the_dict)

def filter_dict_keypatterns(the_dict, keypatterns=[], invert=False):

    def _recursive(the_dict):
            
        new_dict=dict()
        for k,v in the_dict.items():

            if any([re.match(keypattern, k) for keypattern in keypatterns]) ^ invert:

                if isinstance(v, dict):
                    new_dict[k]=_recursive(v)
                
                else:
                    new_dict[k]=v
 
        return new_dict
    
    return _recursive(the_dict)

def filter_dict_keylist(the_dict, keylist=[], invert=False):
    
    def _recursive(the_dict):
        new_dict=dict()
        for k,v in the_dict.items():

            if (k in keylist) ^ invert:

                if isinstance(v, dict):
                    new_dict[k]=_recursive(v)
                
                else:
                    new_dict[k]=v

        return new_dict
    
    return _recursive(the_dict)



def update_dict(orign_dict, mixin_dict, interpret_none_as_val=True, overwrite_if_conflict=True):
    """
    Update origin_dict with values from update_dict. If overwrite is True, values from update_dict will overwrite values from origin_dict
    """

    def _recursive(orign_dict, mixin_dict):
        for k, v in mixin_dict.items():
            # Check if both original[key] and value are dictionaries
            if k in orign_dict and isinstance(orign_dict[k], dict) and isinstance(v, dict):
                # Recurse if both are dictionaries
                orign_dict[k] = _recursive(orign_dict[k], v)

            elif (k not in orign_dict) or ((orign_dict[k] is None) and (not interpret_none_as_val)):
                orign_dict[k] = v

            elif overwrite_if_conflict:
                orign_dict[k] = v

        return orign_dict

    return _recursive(orign_dict, mixin_dict)

def get_keypath_dict(the_dict,key):

    all_keys=get_keys(the_dict, atomic_only=False)
    #print(is_unique(key,all_keys),key,all_keys)
    if not is_unique(key,all_keys): raise KeyError(f"key '{key}' is not unique in the dictionary.")

    def _recursive(the_dict, key):
        for k, v in the_dict.items():
            if k == key:
                return [k]
            elif isinstance(v, dict):
                keypath = _recursive(v, key)
                if keypath:
                    return [k] + keypath
    
    return _recursive(the_dict, key)

def set_keypath(the_dict, keypath, value, inplace=False):

    def _recursive(d, kp, val):
        if len(kp) == 1:
            d[kp[0]] = val
        else:
            if kp[0] not in d:
                d[kp[0]] = {}
            _recursive(d[kp[0]], kp[1:], val)

    if inplace:
        _recursive(the_dict, keypath, value)
        return the_dict
    else:
        new_dict=copy.deepcopy(the_dict)
        _recursive(new_dict, keypath, value)
        return new_dict
    
def nestify_dict_like(the_dict, ref_dict):
    ref_keys = get_keys(ref_dict)
    the_keys = get_keys(the_dict)

    for k in the_keys:

        if not is_unique(k,ref_keys): continue; logging.warning(f"key '{k}' is not unique in the reference dictionary.")
        if not is_unique(k,the_keys): continue; logging.warning(f"key '{k}' is not unique in the dictionary.")

        the_keypath=get_keypath_dict(the_dict, k)
        ref_keypath=get_keypath_dict(ref_dict, k)

        if the_keypath != ref_keypath:

            the_dict=set_keypath(the_dict, ref_keypath, the_dict[k])
            del the_dict[k]

    return the_dict

def update_dict_smart(
        orign_dict,
        mixin_dict,
        interpret_none_as_val=True,
        overwrite_if_conflict=True,
        allow_new_atomic_keys=False,
        allow_new_nested_keys=False,
        ):
    
    mixin_atomic_keys = get_keys(mixin_dict, atomic_only=True)
    mixin_nested_keys = [k for k in get_keys(mixin_dict, atomic_only=False) if k not in mixin_atomic_keys]

    allow_keylist=get_keys(orign_dict, atomic_only=False)
    logging.debug(f"allow_keylist: {allow_keylist}")
    if allow_new_atomic_keys:
        new_atomic_keys=[k for k in mixin_atomic_keys if k not in allow_keylist]
        logging.debug(f"new_atomic_keys: {new_atomic_keys}")
        allow_keylist.extend(new_atomic_keys)

    if allow_new_nested_keys:
        new_nested_keys=[k for k in mixin_nested_keys if k not in allow_keylist]
        logging.debug(f"new_nested_keys: {new_nested_keys}")
        allow_keylist.extend(new_nested_keys)


    mixin_dict=filter_dict_keylist(mixin_dict, keylist=allow_keylist)
    mixin_dict=nestify_dict_like(mixin_dict, orign_dict)

    orign_dict=update_dict(
        orign_dict, 
        mixin_dict, 
        interpret_none_as_val=interpret_none_as_val,
        overwrite_if_conflict=overwrite_if_conflict
        )
    
    return orign_dict