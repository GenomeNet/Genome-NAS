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
num_classes = 4
batch_size= 2


### ResidualBlock definieren ###
# standard layer, which we will need for Residual block 2 times behind each other
def conv3x3(in_channels, out_channels, stride=1): # das ist eben die standard operation/layer, welche im Residual Block 2 mal hintereinander mit Batchnorm und Relu kombiniert wird
    return nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                    stride=stride, padding = 1, bias=False)

# Residual block
class ResidualBlock(nn.Module): # ist eben der ResidualBlock, der eben 2 mal hintereinander diesen 3x3 Layer anwendet und eben identity dazuaddiert (was hier residual = x ist, aber oft durch downsampling auf die richtige dimension gebracht wird)
# ResNet definiert das wirkliche modell, bei dem eben dieser ResidualBlock, der jeweils aus 2 layern besteht mehrmals aneinanderhängen (hier 3 mal) durch makelayer
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
                    conv3x3(in_channels, expanded_channels, stride), # conv3x3() funktion anstatt block() funktion
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
    


# durch make_layer wird jetzt eben 3 mal hintereinander der BasicBlock/Residual Block von oben aneinandergehängt (welche selber wiederrum aus 2 mal 3x3conv() besteht) 
class ResNet(nn.Module):
    def __init__(self, rr_block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        #self.in_channels = 16 # ist glaube ich nur der input channel von den sachen in make_layer, also input_channel von den residual blocks
        self.conv = conv3x3(in_channels = num_classes, out_channels = 16) #  anstatt 1 kommt hier 4 hin, weil ja dann erster Input bei Forward
        # stride ist 1
        # wegen padding=1 haben wir gleich hohe anzahl neuronen, nämlich 10
        self.bn = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.connect_blocks(rr_block, 16, 32, num_blocks[0])# erstes argument "block" entspricht einfach nur BasicBlock() modul (siehe unten)
        # zweites argument 16 output_channels entspricht einfach nur output channel von dem BasicBlock 
        # drittes argument layers[0] bedeutet einfach nur 2, welche wir ja unten bei net_args() definiert haben: in make_layers() kommt ja dann
        # "for i in range(blocks)", d.h. wir haben 2 mal hintereinander diese layers, d.h. BasicBlock beinhaltet immer 2 convolution layers bzw.
        # es werden immer 2 layers übersprungen mit residual
        # als input bekommen wir 16 channels rein, mit jeweils 10 neuronen
        # als output bekommen wir 16 channels, weil so definiert wurde, und 10 neuronen, weil stride=1 (default gelassen)
        self.layer2 = self.connect_blocks(rr_block, 32, 64, num_blocks[0], 2)
        # als output jetzt 32 channels mit 5 neuronen, weil jetzt stride 2 
        self.layer3 = self.connect_blocks(rr_block, 64, 128, num_blocks[1], 2)
        # als output jetzt 64 output channel mit 3 neuronen, weil wieder stride 2
        self.avg_pool = nn.AvgPool1d(3) # weil wir 3 Input Neuronen haben und 1 neuron als output haben (durchschnitt der 3 neuronen)
        self.lstm = nn.LSTM(128, # num_motifs als input channels mit jeweils 3 neuronen: (seq_size-3)/1+1=18; (18-3)/3+1=6
                            128,
                            batch_first=True)
        self.fc1 = nn.Linear(128*3,50) # weil wir 3 neuronen haben
        self.fc2 = nn.Linear(50, num_classes)
       

    def connect_blocks(self, rr_block, in_channels, expanded_channels, num_blocks, stride=1):
        # dann ist nämlich downsample, das was hier 2 zeilen weiter unten steht, nämlich conv3x3() sequentiell ding
        # dieses downsample ist wiederum input argument von layers.append(), also BasicBlock/block
        # in BasicBlock wird downsample layer in forward dann aufgerufen, um residual (identity) zu berechnen
        # die ganze downsample idee brauchen wir glaub nur, weil wenn die anzahl channels abnimmt währen den 2 geskippten layers, dann
        # können wir ja nicht einfach identy addieren, weil ja völlig andere dimension hat, sondern müssen zusätzlich identity downsamplen zuerst
        blocks = []
        blocks.append(rr_block(in_channels, expanded_channels, identity = False)) # anderer block muss über net_args() übergeben werden
        
        for i in range(1, num_blocks):
            blocks.append(rr_block(expanded_channels, expanded_channels, identity = True))
        return nn.Sequential(*blocks) # d.h. zwei layer sequentiell hintereinander 

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
        print('layer3')
        print(x.shape)
        x = self.avg_pool(x)
        x = torch.transpose(x,1, 2)
        
        #CNN expects [batchsize, input_channels, signal_length]
        # lstm expects shape [batchsize, signal_length, number of features]
        output, state = self.lstm(x, prev_state)
        print('out')
        print(output.shape)
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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


# https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278: bei diesem artikel macht er es
# schon genauso wie wir hier: sein shortcut ist halt mein downsampling; er definiert zuerst ResidualBlock, bei dem
# eben identity definiert wird, welcher shortcut hat wenn channels nicht übereinstimmen. aktuell sieht es danach aus, als würde er den anderen residual_block auch einfach 
# als identity definieren, aber später bei ResNetBasicBlock() definiert er self.blocks mit eben 2 layern von ResidualBlocks()
# dieser ResidualBlock wird dann eben in ResNetResidualBlock() eingesetzt;
# diesen setzt er dann wiederum in ResNetBasicBlock() ein, bei denen er jetzt eben self.blocks definiert als 2 layers hintereinander von ResidualBlocks
# dieses expansion brauchen wir nicht, weil wir einfach anstatt 16 output_channels eben 32 und dann 64 angeben, anstatt expansion = 2 einzugeben
