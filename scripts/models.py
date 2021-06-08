import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch




class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        self.conv1 = nn.Conv1d(21, 21, 3)
        self.conv2 = nn.Conv1d(21, 21, 3)
        self.conv3 = nn.Conv1d(21, 21, 3)
        ## decoder layers ##
        self.t_conv1 = nn.ConvTranspose1d(21, 21, 3)
        self.t_conv2 = nn.ConvTranspose1d(21, 21, 3)
        self.t_conv3 = nn.ConvTranspose1d(21, 21, 3)

    def forward(self, x):
        x = F.relu((self.conv1(x)))
        x = F.relu((self.conv2(x)))
        enc = self.conv3(x)
        x = F.relu((enc))
        ## decode ##
        x = F.relu((self.t_conv1(x)))
        x = F.relu((self.t_conv2(x)))
        x = ((self.t_conv3(x)))     
        return x,enc



class ResNet(nn.Module):
    def __init__(self,input_dim, nb_classes):
        super().__init__()
        self.n_feature_maps = 64
        #block1
        self.conv1 = nn.Conv1d(input_dim, self.n_feature_maps, 9, padding=4)
        self.bn1 = nn.BatchNorm1d(self.n_feature_maps)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(self.n_feature_maps, self.n_feature_maps, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(self.n_feature_maps)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(self.n_feature_maps, self.n_feature_maps, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(self.n_feature_maps)
        #residuals
        self.conv4 = nn.Conv1d(input_dim,self.n_feature_maps, 1, padding=0)
        self.bn4 = nn.BatchNorm1d(self.n_feature_maps)
        self.relu4 = nn.ReLU()

        #block2
        self.conv5 = nn.Conv1d(self.n_feature_maps, 2*self.n_feature_maps, 9, padding=4)
        self.bn5 = nn.BatchNorm1d(2*self.n_feature_maps)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv1d(2*self.n_feature_maps, 2*self.n_feature_maps, 5, padding=2)
        self.bn6 = nn.BatchNorm1d(2*self.n_feature_maps)
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv1d(2*self.n_feature_maps, 2*self.n_feature_maps, 3, padding=1)
        self.bn7 = nn.BatchNorm1d(2*self.n_feature_maps)
        #residuals
        self.conv8 = nn.Conv1d(self.n_feature_maps, 2*self.n_feature_maps, 1, padding=0)
        self.bn8 = nn.BatchNorm1d(2*self.n_feature_maps)
        self.relu8 = nn.ReLU()

        #block3
        self.conv9 = nn.Conv1d(2*self.n_feature_maps, 2*self.n_feature_maps, 9, padding=4)
        self.bn9 = nn.BatchNorm1d(2*self.n_feature_maps)
        self.relu9 = nn.ReLU()
        self.conv10 = nn.Conv1d(2*self.n_feature_maps, 2*self.n_feature_maps, 5, padding=2)
        self.bn10 = nn.BatchNorm1d(2*self.n_feature_maps)
        self.relu10 = nn.ReLU()
        self.conv11 = nn.Conv1d(2*self.n_feature_maps, 2*self.n_feature_maps, 3, padding=1)
        self.bn11 = nn.BatchNorm1d(2*self.n_feature_maps)
        #residuals
        self.bn12 = nn.BatchNorm1d(2*self.n_feature_maps)
        self.relu12 = nn.ReLU()
        
        #linear
        self.last = nn.Linear(2*self.n_feature_maps,nb_classes)
               
  
    def forward(self, x):
        x1 = self.relu1(self.bn1(self.conv1(x)))
        x2 = self.relu2(self.bn2(self.conv2(x1)))
        x3 = self.bn3(self.conv3(x2))
        
        x4 = self.bn4(self.conv4(x))
        x = self.relu4(x4+x3)

        x1 = self.relu5(self.bn5(self.conv5(x)))
        x2 = self.relu6(self.bn6(self.conv6(x1)))
        x3 = self.bn7(self.conv7(x2))
        
        x4 = self.bn8(self.conv8(x))
        x = self.relu8(x4+x3)
        
        x1 = self.relu9(self.bn9(self.conv9(x)))
        x2 = self.relu10(self.bn10(self.conv10(x1)))
        x3 = self.bn11(self.conv11(x2))
        
        x4 = self.bn12((x))
        x = self.relu12(x4+x3)
        
        x = x.mean(-1)
        x = self.last(x)
        
        return x