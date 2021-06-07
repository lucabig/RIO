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