#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 07:51:45 2021

@author: amadeu
"""

# train_directory = 'train.mat'
# data_dict = mat73.loadmat('train.mat')
# inputs_small = data_dict["trainxdata"][0:10000]
# targets_small = data_dict["traindata"][0:10000]

# data_dict = scipy.io.loadmat('valid.mat')
# inputs_small_val = data_dict["validxdata"][0:10000]
# targets_small_val = data_dict["validdata"][0:10000]

# data_dict = scipy.io.loadmat('test.mat')
# inputs_small_test = data_dict["testxdata"][0:10000]
# targets_small_test = data_dict["testdata"][0:10000]


#a_file = open("inputs_small_test.pkl", "wb")
#pickle.dump(inputs_small_test, a_file)
#a_file.close()
#b_file = open("targets_small_test.pkl", "wb")
#pickle.dump(targets_small_test, b_file)
#b_file.close()


import pickle

import mat73
import scipy.io


import torch 

from torch.utils.data import Dataset, DataLoader

import numpy as np

#train_input_directory = '/home/amadeu/Desktop/GenomNet_MA/data/inputs_small.pkl'
#train_target_directory = '/home/amadeu/Desktop/GenomNet_MA/data/targets_small.pkl'
#valid_input_directory = '/home/amadeu/Desktop/GenomNet_MA/data/inputs_small_val.pkl'
#valid_target_directory = '/home/amadeu/Desktop/GenomNet_MA/data/targets_small_val.pkl'
#test_input_directory = '/home/amadeu/Desktop/GenomNet_MA/data/inputs_small_test.pkl'
#test_target_directory = '/home/amadeu/Desktop/GenomNet_MA/data/targets_small_test.pkl'
#batch_size = 2

def data_preprocessing(train_directory, valid_directory, test_directory, batch_size):
    
    ### falls auf Cluster ####
    # def data_preprocessing(train_directory, valid_directory, batch_size):
    # muss also auch argparser in train_genomicDARTS.py ändern !!!!!!!!!!!
    # train_directory = '/home/amadeu/Desktop/GenomNet_MA/data/train.mat'
    data_dict_train = mat73.loadmat(train_directory)
    inputs_train = data_dict_train["trainxdata"]
    targets_train = data_dict_train["traindata"]
    # valid_directory = '/home/amadeu/Desktop/GenomNet_MA/data/valid.mat'
    data_dict_val = mat73.loadmat(valid_directory)
    inputs_val = data_dict_val["validxdata"]
    targets_val = data_dict_val["validdata"]
    # test_directory = '/home/amadeu/Desktop/GenomNet_MA/data/test.mat'
    data_dict_test = mat73.loadmat(test_directory)
    inputs_test = data_dict_test["testxdata"]
    targets_test = data_dict_test["testdata"]
  
    
    ### falls auf meinem laptop ###
    #a_file = open(train_input_directory, "rb")
    #inputs_train = pickle.load(a_file)

    #b_file = open(train_target_directory, "rb")
    #targets_train = pickle.load(b_file)
    
    #c_file = open(valid_input_directory, "rb")
    #inputs_val = pickle.load(c_file)

    #d_file = open(valid_target_directory, "rb")
    #targets_val = pickle.load(d_file)
    
    #e_file = open(test_input_directory, "rb")
    #inputs_test = pickle.load(e_file)

    #f_file = open(test_target_directory, "rb")
    #targets_test = pickle.load(f_file)
    
    
    inputs_train, targets_train = torch.Tensor(inputs_train), torch.Tensor(targets_train)
    inputs_train, targets_train = inputs_train.float(), targets_train.float()
    
    # inputs_val, targets_val = np.array(inputs_val), np.array(targets_val)
    inputs_val, targets_val = torch.Tensor(inputs_val), torch.Tensor(targets_val)
    inputs_val, targets_val = inputs_val.float(), targets_val.float()
    
    inputs_test, targets_test = torch.Tensor(inputs_test), torch.Tensor(targets_test)
    inputs_test, targets_test = inputs_test.float(), targets_test.float()


    
    
    class get_data(Dataset):
        def __init__(self,feature,target):
            self.feature = feature
            self.target = target
        def __len__(self):
            return len(self.feature)
        def __getitem__(self,idx):
            item = self.feature[idx]
            label = self.target[idx]
            return item,label
        
    
    train_feat = []
    for batch in inputs_train:
        train_feat.append(batch)
        
    train_targ = []
    for batch in targets_train:
        train_targ.append(batch)
      
        
    val_feat = []
    for batch in inputs_val:
        val_feat.append(batch)
        
    val_targ = []
    for batch in targets_val:
        val_targ.append(batch)
        
        
    test_feat = []
    for batch in inputs_test:
        test_feat.append(batch)
        
    test_targ = []
    for batch in targets_test:
        test_targ.append(batch)
        
        
    train = get_data(train_feat, train_targ)# 
    valid = get_data(val_feat, val_targ)
    test = get_data(test_feat, test_targ)

    
    train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True)#  shuffle ensures random choices of the sequences
    valid_loader = torch.utils.data.DataLoader(valid, batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(valid, batch_size, shuffle=False)

    
    return train_loader, valid_loader, test_loader