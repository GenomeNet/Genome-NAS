#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 11:09:12 2021

@author: amadeu
"""

import data_preprocessing as dp
import run_architectures as run


# get train_loader, valid_loader
train_object, valid_object, num_classes = dp.data_preprocessing(data_file = '/home/amadeu/Desktop/genom_Models/genom_venv/data/trainset.txt', 
          seq_size = 10, representation = 'onehot', model_type = 'CNN', batch_size=2)


# run NCNet with bottleneck blocks
import models.NCNet_bRR_model 
run.run_architecture(models.NCNet_bRR_model.model, models.NCNet_bRR_model.device, models.NCNet_bRR_model.optimizer, models.NCNet_bRR_model.criterion, 3, train_object, valid_object, 2)


# run NCNet with residual blocks
import models.NCNet_RR_model 
run.run_architecture(models.NCNet_RR_model.model, models.NCNet_RR_model.device, models.NCNet_RR_model.optimizer, models.NCNet_RR_model.criterion, 3, train_object, valid_object, 2)


# run DeepVirFinder


# run DanQ