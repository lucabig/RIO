import torch.nn as nn
import torch
import numpy as np
from matplotlib import pyplot as plt
import pickle
import pandas as pd
from os import listdir
from os.path import isfile, join
from collections import defaultdict 
from matplotlib import pyplot as plt
from models import *
from dataset_creation import *



def online_evaluation(data,weights_path,scalers_path,quants_path,length):
    model = ConvAutoencoder()
    model.load_state_dict(torch.load(weights_path))
    with open(quants_path, "rb") as fp:
         quants = pickle.load(fp)
    with open(scalers_path, "rb") as fp:
         scalers = pickle.load(fp)

    


