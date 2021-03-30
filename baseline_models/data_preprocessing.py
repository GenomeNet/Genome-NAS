#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 19:47:29 2020

@author: amadeu
"""
######## data preprocessing #####
import torch
from collections import Counter
import numpy as np
from numpy import array
from torch.utils.data import Dataset,DataLoader

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

        
def data_preprocessing(data_file, seq_size, representation, model_type, batch_size):
    
    def open_data(data_file):
        with open(data_file, 'r') as f:
            text = f.read()
        
        def split(text): 
            return [char for char in text] 
            
        text = split(text)
        
        text = text[:-1]#because last row/observation is empty space
            
        word_counts = Counter(text)# counter object is needed for sorted_vocab bzw. int_to_vacab and vocab_to_int 
        sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
        int_to_vocab = {k: w for k, w in enumerate(sorted_vocab)}
        vocab_to_int = {w: k for k, w in int_to_vocab.items()}
        num_classes = len(int_to_vocab)
        
        int_text = [vocab_to_int[w] for w in text] 
        
        text = np.asarray(int_text)
        
        return text, num_classes, int_to_vocab, vocab_to_int


    def create_sequences(in_text, seq_size, representation, model_type): 
        x = list()
        y = list()
        
        if (representation == 'onehot'):
                int_enc = in_text.reshape(len(in_text), 1)
                in_text = OneHotEncoder(sparse=False).fit_transform(int_enc)
        
                text_len = len(in_text)
                for i in range(text_len):
                    
                    idx = i + seq_size #sequence end
                    
                    if idx > len(in_text)-1: 
                        break
                    if (model_type == 'CNN'): # because CNN, needs one prediction for each sequence (target)
                        feat_seq, target_seq = in_text[i:idx], in_text[idx] # target labels for CNN
                        x.append(feat_seq)
                        y.append(target_seq)
                    if (model_type == 'LSTM'): # because LSTM, needs for each step of the sequence a prediction (target)
                        feat_seq, target_seq = in_text[i:idx], in_text[(i+1):(idx+1)] # target labels for LSTM
                        x.append(feat_seq)
                        y.append(target_seq)
            
        if (representation == 'embedding'):
        
                text_len = len(in_text)
                for i in range(text_len):
                    
                    idx = i + seq_size #sequence end
                    
                    if idx > len(in_text)-1: 
                        break
                    if (model_type == 'CNN'): # because CNN, needs one prediction for each sequence (target)
                        feat_seq, target_seq = in_text[i:idx], in_text[idx] # target labels for CNN
                        x.append(feat_seq)
                        y.append(target_seq)
                    if (model_type == 'LSTM'): # because LSTM, needs for each step of the sequence a prediction (target)
                        feat_seq, target_seq = in_text[i:idx], in_text[(i+1):(idx+1)] # target labels for LSTM
                        x.append(feat_seq)
                        y.append(target_seq)
        return array(x), array(y)



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
    
    text, num_classes, int_to_vocab, vocab_to_int = open_data(data_file)#/home/scheppacha/data/trainset.txt')
    
    x,y = create_sequences(in_text = text, seq_size = seq_size, representation = representation, model_type = model_type)
    
    train_x, train_y = x[0:1500000,:], y[0:1500000]
    
    valid_x,valid_y = x[1500000:2000000,:], y[1500000:2000000]
    
    num_values_batch = batch_size*seq_size # "daten verbrauch" pro batch
  
    num_batches_train = train_y.shape[0] // (num_values_batch)# anzahl minibatches
    end_train = num_values_batch*num_batches_train # macht keinen sinn, weil es bis 6.000.000 geht
    
    num_batches_valid = valid_y.shape[0] // (num_values_batch)# anzahl minibatches
    end_valid = num_values_batch*num_batches_valid
    
    train_x, train_y = train_x[0:end_train,:], train_y[0:end_train]
    valid_x, valid_y = valid_x[0:end_valid,:], valid_y[0:end_valid]
    
    train = get_data(train_x,train_y)# 
    valid = get_data(valid_x, valid_y)
    train_loader = torch.utils.data.DataLoader(train,batch_size,shuffle=True)#  
    valid_loader = torch.utils.data.DataLoader(valid,batch_size,shuffle=True)
    return train_loader, valid_loader, num_classes



