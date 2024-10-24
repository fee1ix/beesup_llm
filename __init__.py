
import os
from .toolkit import *
from .toolkit.setup_utils import *

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


base_directory_config=dict(
    type = 'directory',
)

class BaseDirectory(object):
    """
    Base class for subdirectories inside a lab
    """

    def __init__(self, config=base_directory_config, parent_lab: Optional[Lab]=None):

        

        config=load_dict(config)
        for k, v in config.items(): 
            if v is not None:
                setattr(self, k, v)

        
        if (not hasattr(self, 'parent_lab_path')) and (parent_lab is None):
            self.get_parent_lab()
        
        elif parent_lab:
            self.parent_lab_path=parent_lab.path
    
        # gather the most essential initialisation attributes
        if not hasattr(self, 'type'): 
            raise ValueError("missing type key")
        
        else:
            self.parent_dir_path=f'{self.parent_lab_path}/{self.type}s' # derive the parent directory path from the type (e.g. dataset -> datasets)
            if not os.path.exists(f'{self.parent_dir_path}'):
                os.makedirs(f'{self.parent_dir_path}', exist_ok=False)
        
        if not hasattr(self, 'id'):
            self.get_id()
            self.name=f"{str(self.id).zfill(4)}_{self.type}"
            self.path=f"{self.parent_dir_path}/{self.name}"

        
        if not hasattr(self, 'datetime_init'):
            #self.timestamp_init=datetime.datetime.now(TIMEZONE).strftime('%Y-%m-%d_%H-%M')
            self.datetime_init=datetime.datetime.now(TIMEZONE)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"{self.name.upper()} initialised")
 
    
    def get_parent_lab(self):
        self.parent_lab = Lab()
        self.parent_lab_path = self.parent_lab.path

        return self.parent_lab

    def get_id(self):

        if not hasattr(self, 'id'):

            id_list = get_ids(self.parent_dir_path)
            if id_list: max_id = max(id_list); self.id = max_id + 1
            else: self.id = 1

        return self.id


    def get_config(self):

        key_order=['type', 'id', 'name', 'path', 'parent_dir_path', 'parent_lab_path']
        keys_to_exclude=['parent_lab']

        config = {k: getattr(self, k) for k in key_order if hasattr(self, k)}
        config.update({k: v for k, v in self.__dict__.items() if (k not in key_order) and (k not in keys_to_exclude)})

        return config


    def build(self):

        if not os.path.exists(f'{self.path}'):
            os.makedirs(f'{self.path}', exist_ok=False)

    def __repr__(self):
        return f"{self.name}"






    

    

    






    
    
