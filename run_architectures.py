#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 21:17:41 2020

@author: amadeu
"""

import data_preprocessing as dp
import torch
import train_valid as train_valid
import train_valid_LSTM as train_valid_LSTM


import gc

#
def run_architecture(model, device, optimizer, criterion, epochs, train_loader, valid_loader, batch_size):
        
    for epoch in range(epochs):
        print('epochs {}/{}'.format(epoch+1,epochs))
        if (model = models.NCNet_bRR_model.model | models.NCNet_RR_model.model | models.DanQ.model):
            train_valid_LSTM.Train(model, device, optimizer, criterion, train_loader, valid_loader, batch_size)
            train_valid_LSTM.Valid(model, device, optimizer, criterion, train_loader, valid_loader, batch_size)
        if (model = models.DeepVirFinder_model.model):
            train_valid.Train(model, device, optimizer, criterion, train_loader, valid_loader, batch_size)
            train_valid.Valid(model, device, optimizer, criterion, train_loader, valid_loader, batch_size)
        gc.collect()



