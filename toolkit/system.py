import torch
import random
import logging
import warnings
import numpy as np


def set_seeds(seed=42):
    """
    Set seeds for reproducibility in PyTorch and NumPy.

    This function sets seeds for random number generators in PyTorch and NumPy to ensure
    reproducibility of results.

    Parameters:
        seed (int): Seed value for random number generation. Default is 42.

    Example:
        >>> set_seeds(seed=42)
    """
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)
    np.random.seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 5)
    random.seed(seed + 6)
    logging.info(f'Seed was set to {seed}')

def ignore_warnings(ignore_messages=None):
    """
    Ignore specific warning messages.

    This function allows you to ignore specific warning messages during the execution of
    the program.

    Parameters:
        ignore_messages (str or list, optional): Warning message(s) to ignore. Default is None.

    Example:
        >>> ignore_warnings(ignore_messages=['deprecated', 'module'])

    """
    if ignore_messages:
        if ignore_messages == 'all':
            warnings.filterwarnings('ignore')
        else:
            for warning_msg in ignore_messages:
                warnings.filterwarnings('ignore', message=warning_msg)

import pytz
import datetime
TIMEZONE = pytz.timezone('Europe/Berlin')
TIMESTAMP_FORMAT='%Y-%m-%d_%H-%M-%S'


def get_datetime(a_timestamp=None):
    if a_timestamp is not None:
        return datetime.datetime.strptime(a_timestamp, TIMESTAMP_FORMAT).replace(tzinfo=TIMEZONE)
    else:
        return datetime.datetime.now(TIMEZONE)

def get_timestamp(a_datetime=None):
    if a_datetime is None:
        a_datetime=get_datetime()

    return a_datetime.strftime(TIMESTAMP_FORMAT)
