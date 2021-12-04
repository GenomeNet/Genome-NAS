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

# 2 rr_blocks
# jeder rr_block: conv1d, bn, relu, conv1d, bn
# rest wie bei danQ (also num_motifs, letzter fc layer etc., loss-function)
# bei residual_block immer padding verwenden, damit anzahl neuronen gleicht bleibt (im orginal braucht er padding 1 für conv3x3, damit neuronen gleich bleiben); nur falls channel_size nicht gleich ist, dann shortcut verwenden
# nur durch stride wird ggf. gedownsampled

# da wir nur 2 Residualblocks haben, macht es Sinn erstmal durch conv channelsize auf 320 bringen und dann 2 Residualblocks (außerdem mach er bei ResNet auch vor den blocks erstmal convlayer)
# in den Residualblocks kein downsampling machen, weil er ja schreibt, dass der Rest gleich ist wie bei DanQ (und hier macht er downsampling durch max-pooling mit stride=13)
# normalerweise macht er bei Projection-block immer shortcut mit 1x1 conv umd channelsize zu erhöhen, aber er macht auch immer downsampling mit stride=2, weil auch erster conv layer vom normalen layer stride=2 hat;
# wir machen aber ohne stride=2, weil wir rest wie in DanQ haben wollen und nur durch maxpooling downsamplen


### ResidualBlock definieren ###
def conv_26(in_channels, out_channels, kernel_size=26, stride=1, padding=12): # 
    return nn.Conv1d(in_channels, out_channels, kernel_size, 
                    stride, padding)

def conv_1(in_channels, out_channels, kernel_size=26, stride=1, padding=12): # 
    return nn.Conv1d(in_channels, out_channels, kernel_size, 
                    stride, padding)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Residual block
class ResidualBlock(nn.Module): # 
    def __init__(self, in_channels, expanded_channels, identity = False, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv_26(in_channels, expanded_channels, 26, 1, 12)
        self.bn1 = nn.BatchNorm1d(expanded_channels)
        self.relu = nn.ReLU()
        self.conv2 = conv_26(expanded_channels, expanded_channels, 26, 1, 13)
        self.bn2 = nn.BatchNorm1d(expanded_channels)
        self.identity = identity
        if (self.identity == False): 
            self.shortcut = nn.Sequential(
                    conv_1(in_channels, expanded_channels, 1, 1, 0), 
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
        # print(x.shape)
        # print(residual.shape)
        x += residual
        x = self.relu(x)
        return x 
    


class NCNet_RR(nn.Module):
    def __init__(self, res_block, seq_size, num_classes, batch_size, task):
        super(NCNet_RR, self).__init__()
        self.num_classes = num_classes
        self.num_motifs = 320
        self.batch_size = batch_size
        self.conv1d = nn.Conv1d(4, self.num_motifs, kernel_size=26, stride=1) # with seq_len= 1000 kernel_size=26; seq_len=150 kernelsize=9
        self.seq_size = seq_size
        self.ResidualBlock = res_block
        
        self.bn = nn.BatchNorm1d(self.num_motifs)
        self.relu = nn.ReLU()
        self.block1 = self.ResidualBlock(self.num_motifs, self.num_motifs, identity = True)
        
        self.block2 = self.ResidualBlock(self.num_motifs, self.num_motifs, identity = True)

        # self.block1 = self.connect_blocks(rr_block, 16, 32, num_blocks[0])
        # self.block2 = self.connect_blocks(rr_block, 32, 64, num_blocks[0], 2)      
      
        self.maxpool = nn.MaxPool1d(kernel_size = 13, stride = 13) # seq_len=1000 kernel_size= 13 stride=13; seq_len=150 kernelsize=4 stride=4
        self.dropout = nn.Dropout(p=0.2)
            
        self.lstm = nn.LSTM(self.num_motifs, #
                            self.num_motifs,
                            bidirectional=True)
       
        
    
        aux_num = (((seq_size-26)+1)-13)/13+1
        self.num_neurons = math.floor(aux_num)

        self.dropout_2 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(self.num_motifs*2*self.num_neurons, 925) # seq_len 1000 75neuronen; seq_len 150 35neuronen
        self.fc2 = nn.Linear(925, num_classes)
        
        if task == "TF_bindings":
            
            self.final = nn.Sigmoid()
            
        else:
            
            self.final = nn.Identity()
       

    def connect_blocks(self, res_block, in_channels, expanded_channels, num_blocks, stride=1):
       
        blocks = []
        blocks.append(res_block(in_channels, expanded_channels, identity = False)) 
        
        for i in range(1, num_blocks):
            blocks.append(res_block(expanded_channels, expanded_channels, identity = True))
        return nn.Sequential(*blocks) #

    def forward(self, x, batch_size):
        
        x = self.conv1d(x)
        #print('out') #wegen padding = 1 bleibt output gleich 10 Neuronen
        #print(x.shape)
        x = self.bn(x)
        
        x = self.relu(x)
        #print('bn') # bn bleibt von den dimensionen eh gleich
        #print(x.shape)
        x = self.block1(x)
        x = self.relu(x)
        #print('layer1') 
        #print(x.shape)
        x = self.block2(x)
        #print('layer2')
        #print(x.shape)
      
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
        return (torch.zeros(2, batch_size, self.num_motifs).to(device),
               torch.zeros(2, batch_size, self.num_motifs).to(device))


# rr_block, seq_size, num_classes, num_motifs, batch_size



# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

