#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 14:06:32 2020

@author: amadeu
"""

import csv

import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import Counter
from argparse import Namespace
import pandas as pd

import os

from numpy import array
import torch
import gc
from tqdm import tqdm_notebook as tqdm
from torch.utils.data import Dataset,DataLoader

#num_motifs = 7
#num_classes = 4
#batch_size= 2


### ResidualBlock definieren ###
def conv3x3(in_channels, out_channels, stride=1): # 
    return nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                    stride=stride, padding = 1, bias=False)

# Residual block
class ResidualBlock(nn.Module): # 
    def __init__(self, in_channels, expanded_channels, identity = False, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, expanded_channels, stride)
        self.bn1 = nn.BatchNorm1d(expanded_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(expanded_channels, expanded_channels)
        self.bn2 = nn.BatchNorm1d(expanded_channels)
        self.identity = identity
        if (self.identity == False): 
            self.shortcut = nn.Sequential(
                    conv3x3(in_channels, expanded_channels, stride), 
                    nn.BatchNorm1d(expanded_channels))

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if (self.identity == False):
            residual = self.shortcut(residual)
        x += residual
        x = self.relu(x)
        return x 
    


class NN_class(nn.Module):
    def __init__(self, rr_block, num_blocks, num_classes, batch_size):
        super(NN_class, self).__init__()
        self.conv = conv3x3(in_channels = num_classes, out_channels = 16) #
        self.bn = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.connect_blocks(rr_block, 16, 32, num_blocks[0])
        self.layer2 = self.connect_blocks(rr_block, 32, 64, num_blocks[0], 2)
        self.layer3 = self.connect_blocks(rr_block, 64, 128, num_blocks[1], 2)
      
        self.avg_pool = nn.AvgPool1d(3) # 
        self.lstm = nn.LSTM(128, 
                            128,
                            batch_first=True)
        self.fc1 = nn.Linear(128*3,50) # weil wir 3 neuronen haben
        self.fc2 = nn.Linear(50, num_classes)
       

    def connect_blocks(self, rr_block, in_channels, expanded_channels, num_blocks, stride=1):
       
        blocks = []
        blocks.append(rr_block(in_channels, expanded_channels, identity = False)) 
        
        for i in range(1, num_blocks):
            blocks.append(rr_block(expanded_channels, expanded_channels, identity = True))
        return nn.Sequential(*blocks) #

    def forward(self, x, prev_state):
        x = self.conv(x)
        #print('out') #wegen padding = 1 bleibt output gleich 10 Neuronen
        #print(x.shape)
        x = self.bn(x)
        #print('bn') # bn bleibt von den dimensionen eh gleich
        #print(x.shape)
        x = self.layer1(x)
        #print('layer1') 
        #print(x.shape)
        x = self.layer2(x)
        #print('layer2')
        #print(x.shape)
        x = self.layer3(x)
      
        x = self.avg_pool(x)
        x = torch.transpose(x,1, 2)
        
        #CNN expects [batchsize, input_channels, signal_length]
        # lstm expects shape [batchsize, signal_length, number of features]
        output, state = self.lstm(x, prev_state)
      
        x = torch.reshape(output, (batch_size,3*128)) # weil wir 3 neuronen haben
     
        x = self.fc1(x)
        
        #x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        #print('x4')
        #print(x.shape)
   
        return x, state
    def zero_state(self, batch_size):
        return (torch.zeros(1, batch_size, 128),  # self.num_motifs
               torch.zeros(1, batch_size, 128)) #  self.num_motifs


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net_args = {
    "rr_block": ResidualBlock,
    "num_blocks": [2, 2, 2, 2],
    "num_classes": 4
}
model = ResNet(**net_args)

criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

