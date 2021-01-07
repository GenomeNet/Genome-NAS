#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 21:17:41 2020

@author: amadeu
"""

import data_preprocessing as dp
import DanQ_model as dq
#import train_valid_CNN_LSTM as tv
import torch
from train_valid_DanQ import Train, Valid
import gc

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


#input = torch.randn(3, 5, requires_grad=True)
#target = torch.empty(3, dtype=torch.long).random_(5)
#print(input)
#print(target)