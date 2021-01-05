#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 19:52:23 2020

@author: amadeu
"""

import data_preprocessing as dp
import DeepVirFinder_model as dvf
#import train_valid_CNN_LSTM as tv
import torch
from train_valid_DeepViraFinder import Train, Valid
import gc
from DeepVirFinder_model import model

#from pytorch_model_summary import summary



#
def main():
    epochs = 100
    for epoch in range(epochs):
        print('epochs {}/{}'.format(epoch+1,epochs))
        Train()
        Valid()
        gc.collect()


if __name__ == '__main__':
    main()
    
    
#m = nn.AdaptiveMaxPool1d(1)
#input = torch.randn(1, 12, 8)#1 batch mit 12 channels à 8 neuronen
#output = m(input)

#maxpool = nn.MaxPool1d(kernel_size = 8)
#output2 = maxpool(input)
