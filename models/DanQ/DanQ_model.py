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

import numpy as np 
import pandas as pd 
import os

from numpy import array
import torch
import gc
from tqdm import tqdm_notebook as tqdm
from torch.utils.data import Dataset,DataLoader

num_motifs = 320
num_classes = 4
batch_size= 32

 
   
class DanQ(nn.Module):
    def __init__(self, num_classes, num_motifs):
        super(DanQ,self).__init__()
        self.num_classes = num_classes
        self.num_motifs = num_motifs
        self.conv1d = nn.Conv1d(num_classes,num_motifs,kernel_size=10)# 
        # 
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size = 3, stride = 3)
        #self.dropout = nn.Dropout(p=0.1)
        
        self.lstm = nn.LSTM(320, # als input haben wir hier 2 batches, 6 channels mit jeweils 3 neuronen
                            num_motifs,
                            batch_first=True)
        self.fc1 = nn.Linear(320*30,200) # num_motifs*3
        self.fc2 = nn.Linear(200, num_classes)
        self.sigmoid = nn.Sigmoid()
        #self.fc2 = nn.Linear(4,1)
        
    def forward(self,x,prev_state):
      
        x = self.conv1d(x)
        
        x = self.relu(x)
       
        x = self.maxpool(x
        x = torch.transpose(x, 1, 2) #as CNN expects shape [batchsize, input_channels, signal_length], but now 
        # lstm expects shape [batchsize, signal_length, number of features]
        #print('maxpool')
        #print(x)
        #print(x.shape)
        output, state = self.lstm(x, prev_state)
        print('output')
        print(output)
        print(output.shape)

        x = torch.reshape(output, (batch_size,30*320))
        #x = torch.flatten(x,1)
        #print('x_flat')
        #print(x)
        #print(x.shape) 
        #x = self.dropout(x)
        x = self.fc1(x)
        #print('x4')
        #print(x)
        #print(x.shape)
        #x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x, state
    def zero_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.num_motifs),
               torch.zeros(1, batch_size, self.num_motifs))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DanQ(num_classes, num_motifs).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)



#from modelsummary import summary
#model = DanQ(num_classes=4,num_motifs=6)
#x=torch.zeros((2,4,10))
#state_h = torch.zeros((1,2,6))# für jeden b
#state_c = torch.zeros((1,2,6))
#prev_state = state_h, state_c
#print(summary(model, x, prev_state, show_input=False)) # other output shapes than keras summary, as we receive lstm hiddenstate output instead of lstm output




