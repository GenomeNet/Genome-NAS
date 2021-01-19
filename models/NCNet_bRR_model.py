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


# standard layer, which we will need for Residual block 2 times behind each other
# will use also as shortcut/identity
def conv3x3(in_channels, out_channels, stride=1): # das ist eben die standard operation/layer, welche im Residual Block 2 mal hintereinander mit Batchnorm und Relu kombiniert wird
    return nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                    stride=stride, padding = 1, bias=False) # hier brauchen wir padding, damit selbe dimension bleibt

def conv1x1(in_channels, out_channels, stride=1): # das ist eben die standard operation/layer, welche im Residual Block 2 mal hintereinander mit Batchnorm und Relu kombiniert wird
    return nn.Conv1d(in_channels, out_channels, kernel_size=1, 
                    stride=stride, padding = 0, bias=False) # hier brauchen wir kein padding, damit selbe dimension bleibt


class IdentityBlock(nn.Module): # ist eben der ResidualBlock, der eben 2 mal hintereinander diesen 3x3 Layer anwendet und eben identity dazuaddiert (was hier residual = x ist, aber oft durch downsampling auf die richtige dimension gebracht wird)
# ResNet definiert das wirkliche modell, bei dem eben dieser BasicBlock(=ResidualBlock), der jeweils aus 2 layern besteht mehrmals aneinanderhängen (hier 3 mal) durch makelayer
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
        print('conv1_idb')
        print(x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        print('conv2_idb')
        print(x.shape)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        print('conv3_idb')
        print(x.shape)
        x = self.bn3(x)
        
        x += residual
        x = self.relu(x)
        print('final_idb')
        print(x.shape)
        return x 
    
    
class ProjectionBlock(nn.Module): # ist eben der ResidualBlock, der eben 2 mal hintereinander diesen 3x3 Layer anwendet und eben identity dazuaddiert (was hier residual = x ist, aber oft durch downsampling auf die richtige dimension gebracht wird)
# ResNet definiert das wirkliche modell, bei dem eben dieser BasicBlock(=ResidualBlock), der jeweils aus 2 layern besteht mehrmals aneinanderhängen (hier 3 mal) durch makelayer
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
                conv3x3(in_channels, expanded_channels, stride=2), # conv3x3() funktion anstatt block() funktion
                nn.BatchNorm1d(expanded_channels))
        

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        print('conv1_prb')
        print(x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        print('conv2_prb')
        print(x.shape)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        print('conv3_prb')
        print(x.shape)
        x = self.bn3(x)
        residual = self.shortcut(residual)
       
        print('prb_residual_out')
        print(residual.shape)
        x += residual
        x = self.relu(x)
        print('final_prb')
        print(x.shape)
        return x 


# durch make_layer wird jetzt eben 3 mal hintereinander der BasicBlock/Residual Block von oben aneinandergehängt (welche selber wiederrum aus 2 mal 3x3conv() besteht) 
class ResNet(nn.Module):
    def __init__(self, id_block, pr_block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        #self.in_channels = 16 # ist glaube ich nur der input channel von den sachen in make_layer, also input_channel von den residual blocks
        self.conv = conv3x3(num_classes, 16) # anstatt 1 kommt hier 4 hin, weil ja dann erster Input bei Forward
        self.bn = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.connect_blocks(id_block, pr_block, 16, 8, 32, num_blocks[0])# erstes argument "block" entspricht einfach nur BottleNeckBlock() modul (siehe unten)
        # zweites argument 8 output_channels entspricht einfach nur output channel von dem ersten 1x1 conv, als die dimension, auf die wir reduzieren wollen, danach soll aber wieder auf 16 durch expanded channels gebracht werden
        #  4tes argument layers[0] bedeutet einfach nur 2, welche wir ja unten bei net_args() definiert haben: in make_layers() kommt ja dann
        # "for i in range(blocks)", d.h. wir haben 2 mal hintereinander diese layers, d.h. BasicBlock beinhaltet immer 2 convolution layers bzw.
        # es werden immer 2 layers übersprungen mit residual
        # als input bekommen wir 16 channels rein, mit jeweils 10 neuronen
        # als output bekommen wir 16 channels, weil so definiert wurde, und 10 neuronen, weil stride=1 (default gelassen)
        self.layer2 = self.connect_blocks(id_block, pr_block, 32, 16, 64, num_blocks[0])
        # als output jetzt 32 channels mit 5 neuronen, weil jetzt stride 2 
        self.layer3 = self.connect_blocks(id_block, pr_block, 64, 32, 128, num_blocks[0])
        # als output jetzt 64 output channel mit 3 neuronen, weil wieder stride 2
        self.avg_pool = nn.AvgPool1d(2)
        self.lstm = nn.LSTM(128, # num_motifs als input channels mit jeweils 3 neuronen: (seq_size-3)/1+1=18; (18-3)/3+1=6
                            128,
                            batch_first=True)
        self.fc1 = nn.Linear(128*1,50) #neuronen= 1
        self.fc2 = nn.Linear(50, num_classes)
        #self.sigmoid = nn.Sigmoid()

    def connect_blocks(self, id_block, pr_block, in_channels, reduced_channels, expanded_channels, num_blocks): # blocks ist einfach nur die anzahl an der blocks die hintereinander kommen
        #downsample = None # downsample ist immer "None", es sei denn stride ist nicht 1 oder in_channels ist nicht gleich wie out_channels
        # dann ist nämlich downsample, das was hier 2 zeilen weiter unten steht, nämlich conv3x3() sequentiell ding
        # dieses downsample ist wiederum input argument von layers.append(), also BasicBlock/block
        # in BasicBlock wird downsample layer in forward dann aufgerufen, um residual (identity) zu berechnen
        # die ganze downsample idee brauchen wir glaub nur, weil wenn die anzahl channels abnimmt währen den 2 geskippten layers, dann
        # können wir ja nicht einfach identy addieren, weil ja völlig andere dimension hat, sondern müssen zusätzlich identity downsamplen zuerst
        #if (stride != 1) or (self.in_channels != out_channels):
        #    downsample = nn.Sequential(
        #        conv3x3(self.in_channels, expanded_channels, stride=stride), # conv3x3() funktion anstatt block() funktion
        #        nn.BatchNorm1d(expanded_channels)
        blocks = []
        blocks.append(pr_block(in_channels, reduced_channels, expanded_channels)) # anderer block muss über net_args() übergeben werden
        
        for i in range(1, num_blocks): # geht bei 1 los und nicht bei 0, weil 0 schon 2 zeilen weiter oben definiert wurde
            blocks.append(id_block(reduced_channels, expanded_channels))
        return nn.Sequential(*blocks) # vllt. mit num_blocks versuchen

    def forward(self, x, prev_state):
        x = self.conv(x)
        #print('conv') #wegen padding = 1 bleibt output gleich 10 Neuronen
        #print(out)
        #print(out.shape)
        x = self.bn(x)
        print('bn') # bn bleibt von den dimensionen eh gleich
        print(x.shape)
        x = self.layer1(x)
        print('layer1') 
        print(x.shape)
        x = self.layer2(x)
        print('layer2')
        print(x.shape)
        x = self.layer3(x)
        print('layer3')
        print(x.shape)
        x = self.avg_pool(x)
        #print('avg_pool') # hatten ja vorher 64 channels mit jeweils 3 Neuronen, von diesen 3 neuronen wird der durchschnitt gebildet, was in 64 channels mit 1 neuron führt
        #print(out)
        #print(out.shape)
        x = torch.transpose(x,1, 2)
        #print('avgpool')
        #print(x)
        #print(x.shape)
        #CNN expects [batchsize, input_channels, signal_length]
        # lstm expects shape [batchsize, signal_length, number of features]
        output, state = self.lstm(x, prev_state)
        
        x = torch.reshape(output, (batch_size,1*128))
     
        x = self.fc1(x)
        
        #x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        #print('x4')
        #print(x)
        #print(x.shape)
        #x = self.sigmoid(x)
        #x = x.view(x.size(0), -1)
        #print('flatten')
        #print(out)
        #print(out.shape)
        #x = self.fc(x)
        #print('final_output')
        #print(out)
        #print(out.shape)
        return x, prev_state
    def zero_state(self, batch_size):
        return (torch.zeros(1, batch_size, 128),  # self.num_motifs
               torch.zeros(1, batch_size, 128)) #  self.num_motifs


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


# https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278: bei diesem artikel macht er es
# schon genauso wie wir hier: sein shortcut ist halt mein downsampling; er definiert zuerst ResidualBlock, bei dem
# eben identity definiert wird, welcher shortcut hat wenn channels nicht übereinstimmen. aktuell sieht es danach aus, als würde er den anderen residual_block auch einfach 
# als identity definieren, aber später bei ResNetBasicBlock() definiert er self.blocks mit eben 2 layern von ResidualBlocks()
# dieser ResidualBlock wird dann eben in ResNetResidualBlock() eingesetzt;
# diesen setzt er dann wiederum in ResNetBasicBlock() ein, bei denen er jetzt eben self.blocks definiert als 2 layers hintereinander von ResidualBlocks
# dieses expansion brauchen wir nicht, weil wir einfach anstatt 16 output_channels eben 32 und dann 64 angeben, anstatt expansion = 2 einzugeben
