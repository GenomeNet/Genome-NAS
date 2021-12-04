#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 21:12:02 2021

@author: amadeu
"""

import numpy as np
import torch
import os

# folder = '/home/ascheppa/GenomNet_MA/baseline_models/danQ'
# folder = '/home/amadeu/Desktop/GenomNet_MA/baseline_results/danQ_run1'
            
def convcpu(folder):
    files = os.listdir(folder)
    
    for file in files:
        if 'loss' in file:
            train_loss = np.load(os.path.join(folder, file), allow_pickle=True)
            train_loss = train_loss.astype('float64')
            train_loss = torch.Tensor(train_loss)
            train_loss = train_loss.detach().cpu().numpy()
    
            np.save(os.path.join(folder, file), train_loss)