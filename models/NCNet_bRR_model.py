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

num_classes = 4
batch_size= 2


def conv3x3(in_channels, out_channels, stride=1): #
    return nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                    stride=stride, padding = 1, bias=False) #

def conv1x1(in_channels, out_channels, stride=1): # 
    return nn.Conv1d(in_channels, out_channels, kernel_size=1, 
                    stride=stride, padding = 0, bias=False) #


class IdentityBlock(nn.Module): # 
    def __init__(self, reduced_channels, expanded_channels, stride=1):
        super(IdentityBlock, self).__init__()
        self.conv1 = conv1x1(expanded_channels, reduced_channels, stride)
        self.bn1 = nn.BatchNorm1d(reduced_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(reduced_channels, reduced_channels)
        self.bn2 = nn.BatchNorm1d(reduced_channels)
        self.conv3 = conv1x1(reduced_channels, expanded_channels, stride)
        self.bn3 = nn.BatchNorm1d(expanded_channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
       
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
     
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
       
        x = self.bn3(x)
        
        x += residual
        x = self.relu(x)
      
        return x 
    
    
class ProjectionBlock(nn.Module): # 
    def __init__(self, in_channels, reduced_channels, expanded_channels):
        super(ProjectionBlock, self).__init__()
        self.conv1 = conv1x1(in_channels, reduced_channels, stride=2)
        self.bn1 = nn.BatchNorm1d(reduced_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(reduced_channels, reduced_channels, stride = 1)
        self.bn2 = nn.BatchNorm1d(reduced_channels)
        self.conv3 = conv1x1(reduced_channels, expanded_channels, stride=1)
        self.bn3 = nn.BatchNorm1d(expanded_channels)
        self.shortcut = nn.Sequential(
                conv3x3(in_channels, expanded_channels, stride=2), # 
                nn.BatchNorm1d(expanded_channels))
        

    def forward(self, x):
        residual = x
        x = self.conv1(x)
      
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
       
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
     
        x = self.bn3(x)
        residual = self.shortcut(residual)
      
        x += residual
        x = self.relu(x)
     
        return x 


class ResNet(nn.Module):
    def __init__(self, id_block, pr_block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.conv = conv3x3(num_classes, 16) # 
        self.bn = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.connect_blocks(id_block, pr_block, 16, 8, 32, num_blocks[0])# 
        self.layer2 = self.connect_blocks(id_block, pr_block, 32, 16, 64, num_blocks[0])
        # als output jetzt 64 channels mit 5 neuronen, weil jetzt stride 2 
        self.layer3 = self.connect_blocks(id_block, pr_block, 64, 32, 128, num_blocks[0])
        # als output jetzt 18 output channel mit 3 neuronen, weil wieder stride 2
        self.avg_pool = nn.AvgPool1d(2)
        self.lstm = nn.LSTM(128, # num_motifs als input channels mit jeweils 3 neuronen: (seq_size-3)/1+1=18; (18-3)/3+1=6
                            128,
                            batch_first=True)
        self.fc1 = nn.Linear(128*1,50) #neuronen= 1
        self.fc2 = nn.Linear(50, num_classes)

    def connect_blocks(self, id_block, pr_block, in_channels, reduced_channels, expanded_channels, num_blocks): #
       
        blocks = []
        blocks.append(pr_block(in_channels, reduced_channels, expanded_channels)) # 
        
        for i in range(1, num_blocks): # 
            blocks.append(id_block(reduced_channels, expanded_channels))
        return nn.Sequential(*blocks) # 

    def forward(self, x, prev_state):
        x = self.conv(x)
       
        x = self.bn(x)
    
        x = self.layer1(x)
        
        x = self.layer2(x)
      
        x = self.layer3(x)
    
        x = self.avg_pool(x)
       
        x = torch.transpose(x,1, 2)
       
        output, state = self.lstm(x, prev_state)
        
        x = torch.reshape(output, (batch_size,1*128))
     
        x = self.fc1(x)
        
        x = self.relu(x)
        x = self.fc2(x)
       
        return x, prev_state
    def zero_state(self, batch_size):
        return (torch.zeros(1, batch_size, 128),  # 
               torch.zeros(1, batch_size, 128)) #


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net_args = {
    "id_block": IdentityBlock,
    "pr_block": ProjectionBlock,
    "num_blocks": [2, 2, 2, 2],
    "num_classes": 4
}
model = ResNet(**net_args)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

