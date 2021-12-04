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

import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#trained_pickle_file = '/home/amadeu/Downloads/AMBER-ModelZoo-20211011T133426Z-001/AMBER-ModelZoo/AMBER-Seq.DeepSEA919/hist.pkl'
#import pickle
#with open(trained_pickle_file, 'rb') as f:
#    trained_list = pickle.load(f)


# 8 bottleneck block
# each bottleneck block consists of 3 convlayers, and the first and last one is always with kernelsize=1 and

# da wir jetzt mehrere blocks haben, macht es schon Sinn, die channels step-wise zu erhöhen

def conv_15(in_channels, out_channels, kernel_size=15, stride=1, padding=12): #
    return nn.Conv1d(in_channels, out_channels, kernel_size, 
                    stride, padding, bias=False) #

def conv_1(in_channels, out_channels, kernel_size=1, stride=1, padding=0): # 
    return nn.Conv1d(in_channels, out_channels, kernel_size, 
                    stride, padding, bias=False) #

# 1000-1

class IdentityBlock(nn.Module): 
    # input 160: 40, 160 (input am anfang 160 und am Ende wieder 160, müssen also kein projektion block machen) ; dazwischen wurde auf 40 rezudiert und mit mit dieser channelsize conv ausgeführt
    # input 160: 40, 160 (input am anfang 160 und am Ende wieder 160, müssen also kein projektion block machen) ; dazwischen wurde auf 40 rezudiert und mit mit dieser channelsize conv ausgeführt
    # input 160: 40, 160 (input am anfang 160 und am Ende wieder 160, müssen also kein projektion block machen) ; dazwischen wurde auf 40 rezudiert und mit mit dieser channelsize conv ausgeführt
    # input 160: 40, 160 (input am anfang 160 und am Ende wieder 160, müssen also kein projektion block machen) ; dazwischen wurde auf 40 rezudiert und mit mit dieser channelsize conv ausgeführt
    
    # input 160: 80, 320 (input am anfang 160 und am Ende aber 320, müssen also jetzt projektion block machen !!!) !!!!  ; dazwischen wurde auf 80 rezudiert und mit mit dieser channelsize conv ausgeführt
    # input 320: 80, 320 (input am anfang 320 und am Ende wieder 320, müssen also kein projektion block machen) ; dazwischen wurde auf 80 rezudiert und mit mit dieser channelsize conv ausgeführt
    # input 320: 80, 320 (input am anfang 320 und am Ende wieder 320, müssen also kein projektion block machen) ; dazwischen wurde auf 80 rezudiert und mit mit dieser channelsize conv ausgeführt
    # input 320: 80, 320 (input am anfang 320 und am Ende wieder 320, müssen also kein projektion block machen) ; dazwischen wurde auf 80 rezudiert und mit mit dieser channelsize conv ausgeführt
    def __init__(self, in_channels, reduced_channels, expanded_channels, identity = False, stride=1):
        super(IdentityBlock, self).__init__()
        self.identity = identity
        self.conv1 = conv_1(in_channels, reduced_channels, 1, 1, 0) # von 160 auf 80 reduziert
        self.bn1 = nn.BatchNorm1d(reduced_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_15(reduced_channels, reduced_channels, 15, 1, 7) # bleibt 80 
        self.bn2 = nn.BatchNorm1d(reduced_channels)
        self.conv3 = conv_1(reduced_channels, expanded_channels, 1, 1, 0) # von 80 auf 320 (aber residual eben noch nicht)
        self.bn3 = nn.BatchNorm1d(expanded_channels)
        
        if (self.identity == False): 
            self.shortcut = nn.Sequential(
                    conv_1(in_channels, expanded_channels,  1, 1, 0), 
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
#        print(x.shape)
        if (self.identity == False):
            residual = self.shortcut(residual)
#        print(residual.shape)
        x += residual
        x = self.relu(x)
      
        return x 
    
    
class ProjectionBlock(nn.Module): # 
    def __init__(self, in_channels, reduced_channels, expanded_channels): # 40, 160
        super(ProjectionBlock, self).__init__()
        self.conv1 = conv_1(in_channels, reduced_channels, stride=2)
        # self.conv1 = conv_1(in_channels, reduced_channels, 1, 1, 0)
        
        self.bn1 = nn.BatchNorm1d(reduced_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_15(reduced_channels, reduced_channels, stride = 1)
        # self.conv2 = conv_15(reduced_channels, reduced_channels, 15, 1, 7)

        self.bn2 = nn.BatchNorm1d(reduced_channels)
        self.conv3 = conv_1(reduced_channels, expanded_channels, stride=1)
        # self.conv3 = conv_1(reduced_channels, expanded_channels, 1, 1, 0)

        self.bn3 = nn.BatchNorm1d(expanded_channels)
        self.shortcut = nn.Sequential(
                conv_15(in_channels, expanded_channels, stride=2), # 
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


class NCNet_bRR(nn.Module):
    def __init__(self, id_block, pr_block, seq_size, num_classes, batch_size, task):
        super(NCNet_bRR, self).__init__()
        self.num_classes = num_classes
        self.num_motifs = 320
        self.batch_size = batch_size
        self.conv1d = nn.Conv1d(4, 160, kernel_size=26, stride=1)
        
        self.bn = nn.BatchNorm1d(160)
        self.seq_size = seq_size

        self.relu = nn.ReLU(inplace=True)
        # self.layer1 = self.connect_blocks(id_block, pr_block, 16, 8, 32, num_blocks[0])# 
        # self.layer2 = self.connect_blocks(id_block, pr_block, 32, 16, 64, num_blocks[0])
        self.bottleneck1 = id_block(160, 40, 160, identity = True)
        self.bottleneck2 = id_block(160, 40, 160, identity = True)
        
        self.bottleneck3 = id_block(160, 40, 160, identity = True)
        self.bottleneck4 = id_block(160, 40, 160, identity = True)
        
        self.bottleneck5 = id_block(160, 80, 320, identity = False)
        self.bottleneck6 = id_block(320, 80, 320, identity = True)
        
        self.bottleneck7 = id_block(320, 80, 320, identity = True)
        self.bottleneck8 = id_block(320, 80, 320, identity = True)

        self.num_classes = num_classes
        self.batch_size = batch_size   
        
        self.relu = nn.ReLU()   
      
        self.maxpool = nn.MaxPool1d(kernel_size = 13, stride = 13) # seq_len=1000 kernel_size= 13 stride=13; seq_len=150 kernelsize=4 stride=4
        self.dropout = nn.Dropout(p=0.2)
            
        self.lstm = nn.LSTM(320, #
                            320,
                            bidirectional=True)
    
        aux_num = (((seq_size-26)+1)-13)/13+1
        self.num_neurons = math.floor(aux_num)

        self.dropout_2 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(320*2*self.num_neurons, 925) # seq_len 1000 75neuronen; seq_len 150 35neuronen
        self.fc2 = nn.Linear(925, num_classes)
        
        if task == "TF_bindings":
            
            self.final = nn.Sigmoid()
            
        else:
            
            self.final = nn.Identity()
        
        
        
    def connect_blocks(self, id_block, pr_block, in_channels, reduced_channels, expanded_channels, num_blocks): #
       
        blocks = []
        blocks.append(pr_block(in_channels, reduced_channels, expanded_channels)) # 
        
        for i in range(1, num_blocks): # 
            blocks.append(id_block(reduced_channels, expanded_channels))
        return nn.Sequential(*blocks) # 


    def forward(self, x, batch_size):
        
        x = self.conv1d(x)
        x = self.bn(x)
        
        x = self.relu(x)
        
        x = self.bottleneck1(x)
        x = self.relu(x)
        
        x = self.bottleneck2(x)
        x = self.relu(x)

        x = self.bottleneck3(x)
        x = self.relu(x)

        x = self.bottleneck4(x)
        x = self.relu(x)

        x = self.bottleneck5(x)
        x = self.relu(x)

        x = self.bottleneck6(x)
        x = self.relu(x)

        x = self.bottleneck7(x)
        x = self.relu(x)
        
        x = self.bottleneck8(x)
        x = self.relu(x)

        x = self.maxpool(x)
        
        x = self.dropout(x)
        
        h_0, c_0 = self.zero_state(batch_size)
        # print(x.shape) # [2,320,75]

        
        # x = torch.transpose(x, 1, 2)
        x = x.permute(2,0,1)

        #CNN expects [batchsize, input_channels, signal_length]
        # lstm expects shape [batchsize, signal_length, number of features]
        # print(x.shape)
        #print(prev_state.shape)

        output, state = self.lstm(x, (h_0, c_0))
        
        # print(x.shape) # [75,2,320]
        x = output.permute(1,0,2)
        # print(x.shape) # [2,75,640]
        
        x = torch.flatten(x, start_dim= 1) 
        x = self.dropout_2(x)


        #x = torch.reshape(output, (self.batch_size,self.num_neurons*self.num_motifs)) # seqlen=1000 75 neuronen; seq_len=150 35 neuronen
        #x = torch.flatten(x,1)

        x = self.fc1(x)
      
        x = self.relu(x)
        
        x = self.fc2(x)
        
        x = self.final(x)
        
        return x
    
    def zero_state(self, batch_size):
        return (torch.zeros(2, batch_size, 320).to(device),
               torch.zeros(2, batch_size, 320).to(device))

