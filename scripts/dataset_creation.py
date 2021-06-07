import numpy as np
from numpy import nan
import pickle
import pandas as pd
from os import listdir
from os.path import isfile, join
from collections import defaultdict 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from utils import *
from torch_utils import *


def create_datasets(class_f=4,stride=5,chunk_size=20,scale='minmax',b=100):
    #read full training data
    data,fields = read_full_data()

    # select only healthy data
    data_h,kk = extract_healthy(data)

    #select only faulty data 
    data_u,kkf = extract_unhealthy(data,class_f)

    # nan filling and field selection
    f_data_h = fill_nans_fields(data_h,kk,fields)
    f_data_u = fill_nans_fields(data_u,kkf,fields)

    #normalize
    norm_h,scaler = normalize(f_data_h,type_scaling=scale)
    norm_u,scaler = normalize(f_data_u,scaler_saved=scaler,type_scaling=scale)

    #Chunk
    train_h = chunking(norm_h,stride=stride,chunk_size=chunk_size)
    train_u = chunking(norm_u,stride=stride,chunk_size=chunk_size)

    #torch datasets and loaders
    dataset_train, train_loader, valid_loader,train_idx,valid_idx = train_test_split(train_h, batch_size=b, val_size=0.35)
    dataset_test, test_loader, valid_test_loader,test_idx,valid_test_idx = train_test_split(train_u, batch_size=b, val_size=0)

    return {'train': (dataset_train, train_loader,valid_loader,train_idx,valid_idx), 'test': (dataset_test, test_loader, valid_test_loader, test_idx,valid_test_idx)}


if __name__ == "__main__":
    my_dict = create_datasets()