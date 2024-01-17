import sys

sys.path.extend([".", ".."])
import os
import random
import regex as re
import pickle
import config
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing
import json
import javalang
import time
from tqdm import *

print('Seeding everything...')
seed = 666
random.seed(seed)  # Python random module.
np.random.seed(seed)  # Numpy module.
torch.manual_seed(seed)  # Torch CPU random seed module.
torch.cuda.manual_seed(seed)  # Torch GPU random seed module.
torch.cuda.manual_seed_all(seed)  # Torch multi-GPU random seed module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['PYTHONHASHSEED'] = str(seed)

print('Seeding Finished')
# Device configuration
use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
