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


def create_datasets_ad(class_f=4,stride=5,chunk_size=20,scale='minmax',b=100):
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
    norm_h,scaler_h = normalize(f_data_h,type_scaling=scale)
    norm_u,_ = normalize(f_data_u,scaler_saved=scaler_h,type_scaling=scale)

    #Chunk
    train_h = chunking(norm_h,stride=stride,chunk_size=chunk_size)
    train_u = chunking(norm_u,stride=stride,chunk_size=chunk_size)

    #torch datasets and loaders
    dataset_train, train_loader, valid_loader,train_idx,valid_idx = train_test_split(train_h, batch_size=b, val_size=0.35)
    dataset_test, test_loader, valid_test_loader,test_idx,valid_test_idx = train_test_split(train_u, batch_size=b, val_size=0)

    return {'train_chunked': (dataset_train, train_loader,valid_loader,train_idx,valid_idx), 
    'test_chunked': (dataset_test, test_loader, valid_test_loader, test_idx,valid_test_idx), 
    'scaler': scaler_h,
    'train_long': norm_h,
    'test_long': norm_u}



def create_datasets_class(stride=5,chunk_size=20,scale='minmax',b=100):
    #read full training data
    data,fields = read_full_data()


    data_h,kk = extract_healthy(data)
    f_data_h = fill_nans_fields(data_h,kk,fields)
    norm_h,scaler_h = normalize(f_data_h,type_scaling=scale)
    train_h = chunking(norm_h,stride=stride,chunk_size=chunk_size)
    labels_h = np.zeros([train_h.shape[0]])

    #select only faulty data
    train_u_total = np.zeros([1,len(scaler_h),chunk_size])
    labels_u_total = np.zeros(1)
    for i in range(1,10): 
        try:
            data_u,kkf = extract_unhealthy(data,i)
            f_data_u = fill_nans_fields(data_u,kkf,fields)
            norm_u,_ = normalize(f_data_u,scaler_saved=scaler_h,type_scaling=scale)
            train_u = chunking(norm_u,stride=stride,chunk_size=chunk_size)
            labels_u = i*np.ones([train_u.shape[0]])
            train_u_total = np.concatenate([train_u_total,train_u])
            labels_u_total = np.concatenate([labels_u_total,labels_u])
        except IndexError:
            print('Faulty data extraction terminated')
    train_u_total = train_u_total[1:,:,:]
    labels_u_total = labels_u_total[1:] 

    train = np.concatenate([train_h,train_u_total],axis=0)
    labels = np.concatenate([labels_h,labels_u_total],axis=0) 

    #torch datasets and loaders
    dataset_train, train_loader, valid_loader,train_idx,valid_idx = train_test_split(train,labels, batch_size=b, val_size=0.35)

    return {'train_chunked': (dataset_train, train_loader,valid_loader,train_idx,valid_idx), 
    'scaler': scaler_h,
    'train_long': norm_h}


if __name__ == "__main__":
    my_dict = create_datasets_class()