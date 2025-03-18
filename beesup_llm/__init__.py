import os
import sys
import logging
import warnings
from typing import Optional, Union


def get_labhandler():
    """Returns an instance of Labhandler if it exists in sys.modules, else None."""
    if 'labtools.labhandler' in sys.modules:
        module = sys.modules['labtools.labhandler']
        return getattr(module, 'Labhandler', None) if hasattr(module, 'Labhandler') else None
    return None

def _isinstance(the_object, the_class):
    """
    Custom implementation of isinstance() to check if an object is an instance 
    of a given class or its subclasses. This function compares the method 
    resolution order (MRO) of the object's class and the target class.
    Args:
        the_object (object): The object to check.
        the_class (type): The class to compare against.
    Returns:
        bool: True if the object is an instance of the class or its subclasses, 
        False otherwise.
    """

    object_parts=the_object.__class__.mro()[:-1]
    class_parts=the_class.mro()[:-1]

    for object_part in object_parts:
        for class_part in class_parts:
            if str(object_part)==str(class_part):
                return True
            
    return False
