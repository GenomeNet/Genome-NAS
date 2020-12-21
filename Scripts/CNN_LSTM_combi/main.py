#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 21:17:41 2020

@author: amadeu
"""

import data_preprocessing as dp
import CNN_LSTM_comb as cl
#import train_valid_CNN_LSTM as tv
import torch
from train_valid_CNN_LSTM import Train, Valid
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
