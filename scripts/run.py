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




def train(n_epochs=50):
    model = ConvAutoencoder()
    print(model)
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0)

    my_dict = create_datasets(class_f=4)
    train_loader=my_dict['train'][1]
    valid_loader=my_dict['train'][2]

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



if __name__ == "__main__":
    train()

