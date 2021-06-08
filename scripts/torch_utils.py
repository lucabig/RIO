import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import numpy as np



class dataset_ad(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.sensor_num = dataset.shape[1]
        self.time_steps = dataset.shape[2]
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.dataset[idx,:,:].astype('float32')
        data = torch.from_numpy(data)
        return data


class dataset_class(Dataset):
    def __init__(self, dataset,labels):
        self.dataset = dataset
        self.labels = labels
        self.sensor_num = dataset.shape[1]
        self.time_steps = dataset.shape[2]
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.dataset[idx,:,:].astype('float32')
        data = torch.from_numpy(data)
        label = self.labels[idx]
        label = torch.tensor(label).long()
        return data,label


def train_test_split(data,labels=None,batch_size=100, val_size=0.35):
    if np.any(labels == None)==True:
        dataset_train = dataset_ad(data)
    else:
        dataset_train = dataset_class(data,labels)
    valid_size = 0.35
    num_train = len(dataset_train)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
        sampler=train_sampler, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, 
        sampler=valid_sampler, num_workers=0)
    return dataset_train, train_loader, valid_loader, train_idx,valid_idx
 