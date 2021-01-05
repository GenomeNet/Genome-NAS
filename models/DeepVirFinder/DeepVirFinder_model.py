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
batch_size = 32 #from data_preprocessing import batch_size
num_motifs = 300
num_classes = 4
 
   
class DeepVirFinder(nn.Module):
    def __init__(self):
        super(DeepVirFinder,self).__init__()
        self.conv1d = nn.Conv1d(4,num_motifs,kernel_size=10)# 4 input channels, für jeden buchstaben, also ein channel und lassen 5 kernels/motifs darüber laufen
        # deswegen haben wir nach dem flatten/x.view(-1) 6*2 neuronen 
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool1d(kernel_size = 91)
        self.globalmaxpool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(num_motifs,num_motifs)   # 
        self.fc2 = nn.Linear(num_motifs,num_classes)
        self.sigmoid = nn.Sigmoid()
        #self.fc2 = nn.Linear(4,1)
        
    def forward(self,x):
        #print('x')
        #print(x)
        #print(x.shape)
        x = self.conv1d(x)
        #print('x1')
        #print(x)
        #print(x.shape)
        x = self.relu(x)
        #print('relu')
        #print(x)
        #print(x.shape)
        x = self.globalmaxpool(x)
        x = torch.reshape(x, (batch_size, num_motifs))
        # here no flatting of channels

        #x = x.view(-1,6*2)# in order to flatten, the 6 depth channel tensor and 2 neurons; (if we have 2 tensors (because 2 batches) with each having 6 channels, we receive a vector with 12 elements)
        #print('xpool')
        #print(x)
        #print(x.shape) # [2,5,1] also Mx1 wie im Paper, also für jeden kernel/motif, einen wert
        x = self.dropout(x)
        x = self.fc1(x)
        #print('dense_1')
        #print(x)
        #print(x.shape)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        #x = self.fc2(x)
        #print('x5')
        #print(x)
        #print(x.shape)
        return x



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepVirFinder().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)



##
#from modelsummary import summary
#model = DeepVirFinder()
#print(summary(model, torch.zeros((32,4,100)), show_input=False))








































