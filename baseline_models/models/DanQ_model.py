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
import math


from numpy import array
import torch
import gc
from tqdm import tqdm_notebook as tqdm
from torch.utils.data import Dataset,DataLoader




class NN_class(nn.Module):
    def __init__(self,num_classes, num_motifs, batch_size, seq_size):
        super(NN_class,self).__init__()
        self.num_classes = num_classes
        self.num_motifs = num_motifs
        self.batch_size = batch_size
        self.conv1d = nn.Conv1d(num_classes,num_motifs,kernel_size=9, stride=1) # with seq_len= 1000 kernel_size=26; seq_len=150 kernelsize=9
        self.seq_size = seq_size
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size = 4, stride = 4) # seq_len=1000 kernel_size= 13 stride=13; seq_len=150 kernelsize=4 stride=4
        self.dropout = nn.Dropout(p=0.1)
        
        self.lstm = nn.LSTM(num_motifs, #
                            num_motifs,
                            batch_first=True)
        aux_num = (((seq_size-9)+1)-4)/4+1
        self.num_neurons = math.floor(aux_num)
        self.fc1 = nn.Linear(num_motifs*self.num_neurons,300) # seq_len 1000 75neuronen; seq_len 150 35neuronen
        self.fc2 = nn.Linear(300, num_classes)
       
        
    def forward(self,x,prev_state):
      
        x = self.conv1d(x) 
        
        x = self.relu(x)
       
        x = self.maxpool(x) 
        x = torch.transpose(x,1, 2)
        #CNN expects [batchsize, input_channels, signal_length]
        # lstm expects shape [batchsize, signal_length, number of features]
     
        x = self.dropout(x)
        
        output, state = self.lstm(x, prev_state)
       
        x = self.dropout(x)
        
        x = torch.reshape(output, (self.batch_size,self.num_neurons*self.num_motifs)) # seqlen=1000 75 neuronen; seq_len=150 35 neuronen
        #x = torch.flatten(x,1)
      
        x = self.fc1(x)
      
        x = self.relu(x)
        x = self.fc2(x)
        return x, state
    
    def zero_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.num_motifs),
               torch.zeros(1, batch_size, self.num_motifs))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = DanQ(num_classes, num_motifs, batch_size).to(device)
#model = DanQ()

criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)



#from modelsummary import summary
#model = DanQ(num_classes=4,num_motifs=6)
#x=torch.zeros((2,4,10))
#state_h = torch.zeros((1,2,6))# für jeden b
#state_c = torch.zeros((1,2,6))
#prev_state = state_h, state_c
#print(summary(model, x, prev_state, show_input=False)) # other output shapes than keras summary, as we receive lstm hiddenstate output instead of lstm output




