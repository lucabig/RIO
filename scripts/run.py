import torch.nn as nn
import torch
import pickle
import pandas as pd
from os import listdir
from os.path import isfile, join
from collections import defaultdict 
from matplotlib import pyplot as plt
from models import *
from dataset_creation import *




def train(dataset_train, dataset_test,train_idx,valid_idx,n_epochs=500):
    model = ConvAutoencoder()
    print(model)
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0)
    valid_loss_min = np.Inf 
    for epoch in range(1, n_epochs+1):
        train_loss = 0.0
        valid_loss = 0.0    
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            outputs, _ = model(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)
        model.eval()
        for data in valid_loader:
            output, _ = model(data)
            loss = criterion(output, data)
            valid_loss += loss.item()*data.size(0)
        
        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(valid_loader.sampler)
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))
        
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), 'C://Users//lbg//OneDrive - CSEM S.A//Bureau//RIO_Data_Challenge//New_Model//Weights//CNNAE_May.pt')
            valid_loss_min = valid_loss
    return model

def test(model, dataset_train, dataset_test,train_idx,valid_idx):
    model.load_state_dict(torch.load('C://Users//lbg//OneDrive - CSEM S.A//Bureau//RIO_Data_Challenge//New_Model//Weights//CNNAE_May.pt'))
    ### val data 
    print('validation scores..')
    model.eval()
    with torch.no_grad():
        valid_data = torch.Tensor(dataset_train.dataset[valid_idx,:,:])
        test_data = torch.Tensor(dataset_test.dataset)
        diff = torch.abs(model(valid_data)[0].detach()-valid_data)
        diff_sum = torch.sum(diff,axis=2).numpy()
    #    diff_sum = (diff_sum/torch.abs(valid_data.mean(-1)).mean(0)).numpy()
        quants = np.array([1.*np.quantile(diff_sum[:,i], 0.99) for i in range(diff_sum.shape[1])])
    tra = diff_sum
    for i in range(diff_sum.shape[1]):
        a = np.sum(tra[:,i]>quants[i])/tra.shape[0]
        if a > 0.2:
            print()
            print(m_subset[i])
            print()
        print(a)
    print()
    print()
    ### train data
    print('train scores..')
    model.eval()
    with torch.no_grad():
        train_data = torch.Tensor(dataset_train.dataset[train_idx,:,:])
        diff = torch.abs(model(train_data)[0].detach()-train_data)
        diff_sum = torch.sum(diff,axis=2).numpy()
    #    diff_sum = (diff_sum/torch.abs(valid_data.mean(-1)).mean(0)).numpy()
    tra = diff_sum
    for i in range(diff_sum.shape[1]):
        a = np.sum(tra[:,i]>quants[i])/tra.shape[0]
        if a > 0.2:
            print()
            print(m_subset[i])
            print()
        print(a)
    print()
    print()
    ### test data
    print('test scores..')
    model.eval()
    with torch.no_grad():
        diff = torch.abs(model(test_data)[0].detach()-test_data)
        diff_sum = torch.sum(diff,axis=2).numpy()
    #    diff_sum = (diff_sum/torch.abs(valid_data.mean(-1)).mean(0)).numpy()
    tra = diff_sum
    for i in range(diff_sum.shape[1]):
        a = np.sum(tra[:,i]>quants[i])/tra.shape[0]
        if a > 0.2:
            print()
            print(m_subset[i])
        print(a)
    return quants



if __name__ == "__main__":
    my_dict = create_datasets(class_f=4,stride=5,chunk_size=10,scale='minmax',b=100)
    train_loader=my_dict['train'][1]
    valid_loader=my_dict['train'][2]
    train_idx=my_dict['train'][3]
    valid_idx=my_dict['train'][4]
    dataset_train=my_dict['train'][0]
    dataset_test=my_dict['test'][0]
    #model = ConvAutoencoder()
    model = train(dataset_train, dataset_test,train_idx,valid_idx,n_epochs=1000)
    quants = test(model, dataset_train, dataset_test,train_idx,valid_idx)

