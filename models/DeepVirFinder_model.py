#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 12:54:39 2020

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
#from pytorch_model_summary import summary
batch_size = 2 #from data_preprocessing import batch_size
num_motifs = 64
num_classes = 4
 
   
class DeepVirFinder(nn.Module):
    def __init__(self):
        super(DeepVirFinder,self).__init__()
        self.conv1d = nn.Conv1d(4,num_motifs,kernel_size=3)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool1d(kernel_size = 91)
        self.globalmaxpool = nn.AdaptiveMaxPool1d(3)
        self.dropout = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(num_motifs*3,num_motifs)   # 
        self.fc2 = nn.Linear(num_motifs,num_classes)
        
    def forward(self,x):
        #print('x')
        #print(x)
        #print(x.shape)
        x = self.conv1d(x)# results in 91 neurons and 6 channels
        #print('x1')
        #print(x)
        #print(x.shape)
        x = self.relu(x)
        #print('relu')
        #print(x)
        #print(x.shape)
        x = self.globalmaxpool(x) # results in 1 neuron and 64 channels
        #print('globalmaxpool')
        #print(x)
        #print(x.shape)
        #x = torch.reshape(x, (batch_size, num_motifs))
        x = x.view(x.size(0), -1)
        #print('after globalmaxpool')
        #print(x)
        #print(x.shape)
        x = self.dropout(x)
        x = self.fc1(x)
        #print('dense_1')
        #print(x)
        #print(x.shape)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepVirFinder().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)



##
#from modelsummary import summary
#model = DeepVirFinder()
#print(summary(model, torch.zeros((32,4,100)), show_input=False)) # as CNN expects shape of [batchsize, input_channel, signal_length]







































