#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 18:53:34 2021

@author: amadeu
"""


######## data preprocessing #####

import torch
from collections import Counter
import numpy as np
from numpy import array
from torch.utils.data import Dataset, DataLoader

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import re

import os

# train_directory = '/home/amadeu/Downloads/genomicData/train' # muss in arg.parser übergeben werden
# valid_directory = '/home/amadeu/Downloads/genomicData/validation'
# num_files = 3
# seq_size = 15
# batch_size = 2


def data_preprocessing(train_directory, valid_directory, num_files, seq_size, batch_size, next_character):
    
    def open_data(train_directory, valid_directory, num_files):
        
        trainfolder_list, validfolder_list = os.listdir(train_directory), os.listdir(valid_directory)

        trainmainfold, validmainfold = ' ', ' '
        new_instance = '/'
        
        for i in range(num_files): # for each folder
            
            trainsubfold = trainfolder_list[i] # würde nicht aufgehen, wenn validfolder_list weniger ordner hat
            trainsubfold = os.path.join(train_directory, trainsubfold) 
            trainlines = open(trainsubfold, 'r').readlines()
            for idx, _ in enumerate(trainlines): # for each line of a folder: delete a line, which begins with '>' or ';'
                if (trainlines[idx][0] == '>' or trainlines[idx][0] == ';'):
                    trainlines[idx] = new_instance
            
            trainlines = ''.join(trainlines) # convert to string
            trainlines = trainlines.replace('\n','')
            trainlines = re.sub('[^/ATCG]', '/', trainlines)
            trainmainfold = trainmainfold + trainlines
            
            
            validsubfold = validfolder_list[i] # würde nicht aufgehen, wenn validfolder_list weniger ordner hat
            validsubfold = os.path.join(valid_directory, validsubfold)  
            validlines = open(validsubfold, 'r').readlines()
                  
            for idx, _ in enumerate(validlines): # for each line of a folder: delete a line, which begins with '>' or ';'
                if (validlines[idx][0] == '>' or validlines[idx][0] == ';'):
                    validlines[idx] = new_instance
            
            validlines = ''.join(validlines) # convert to string
            validlines = validlines.replace('\n','')
            validlines = re.sub('[^/ATCG]', '/', validlines)
            validmainfold = validmainfold + validlines

        
        traintext, validtext = list(trainmainfold), list(validmainfold)
                
        traintext.remove(traintext[0])#because first row/observation is empty space
        validtext.remove(validtext[0])#because first row/observation is empty space

                    
        word_counts = Counter(traintext)# counter object is needed for sorted_vocab bzw. int_to_vacab and vocab_to_int 
        sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
        int_to_vocab = {k: w for k, w in enumerate(sorted_vocab)}
        vocab_to_int = {w: k for k, w in int_to_vocab.items()}
        num_classes = len(int_to_vocab)-1
                
        traintext, validtext = [vocab_to_int[w] for w in traintext], [vocab_to_int[w] for w in validtext]
        
        traintext, validtext = np.asarray(traintext), np.asarray(validtext)
        
        #new_instance_feat = np.count_nonzero(feat_seq[:,4]) # Häufigkeit von "/", was als integer 4 ist
        #new_instance_target = np.count_nonzero(target_seq[4])
        
        return traintext, validtext, num_classes, int_to_vocab, vocab_to_int


    def create_sequences(traintext, validtext, seq_size, next_character): 
        
        train_input, train_target = list(), list()
        valid_input, valid_target = list(), list()
        
        train_int_enc, valid_int_enc = traintext.reshape(len(traintext), 1), validtext.reshape(len(validtext), 1)
        traintext, validtext = OneHotEncoder(sparse=False).fit_transform(train_int_enc), OneHotEncoder(sparse=False).fit_transform(valid_int_enc)

        train_len, valid_len = len(traintext), len(validtext)
        
        if train_len < valid_len:
            leng = train_len
        else:
            leng = valid_len
        
        for i in range(leng):
                     
            idx = i + seq_size #sequence_end
            
            if idx > leng-1: 
                break
            
            if next_character == True:
                feat_seq_train, target_seq_train = traintext[i:idx], traintext[idx]
                feat_seq_train = np.transpose(feat_seq_train)
                # delete the sequence if there is a "/" in the sequence (which is in column 4) skip the sequence 
                # "/" stands for next sample
                #new_instance_feat_train, new_instance_target_train = np.count_nonzero(feat_seq_train[:,4]), np.count_nonzero(target_seq_train[4]) 
                new_instance_feat_train, new_instance_target_train = np.count_nonzero(feat_seq_train[4,:]), np.count_nonzero(target_seq_train[4]) 

                
                feat_seq_valid, target_seq_valid = validtext[i:idx], validtext[idx] # target labels for CNN
                feat_seq_valid = np.transpose(feat_seq_valid)
                #new_instance_feat_valid, new_instance_target_valid = np.count_nonzero(feat_seq_valid[:,4]), np.count_nonzero(target_seq_valid[4]) # Häufigkeit von "/", was als integer 4 ist
                new_instance_feat_valid, new_instance_target_valid = np.count_nonzero(feat_seq_valid[4,:]), np.count_nonzero(target_seq_valid[4]) # Häufigkeit von "/", was als integer 4 ist

            else:
                feat_seq_train, target_seq_train = traintext[i:idx], traintext[i+1:idx+1] 
                # delete the sequence if there is a "/" in the sequence (which is in column 4) skip the sequence 
                # "/" stands for next sample
                new_instance_feat_train, new_instance_target_train = np.count_nonzero(feat_seq_train[:,4]), np.count_nonzero(target_seq_train[:,4]) 
                
                feat_seq_valid, target_seq_valid = validtext[i:idx], validtext[i+1:idx+1] # target labels for CNN
                new_instance_feat_valid, new_instance_target_valid = np.count_nonzero(feat_seq_valid[:,4]), np.count_nonzero(target_seq_valid[:,4]) 
                
            # if there is a new sample, go to next prediction
            if (new_instance_feat_train >= 1 or new_instance_target_train >=1 or new_instance_feat_valid >= 1 or new_instance_target_valid >= 1):
                continue
                        
            # delete last column
            if next_character == True:
                # feat_seq_train, target_seq_train = np.delete(feat_seq_train, -1, axis=1), np.delete(target_seq_train, -1) 
                # feat_seq_valid, target_seq_valid = np.delete(feat_seq_valid, -1, axis=1), np.delete(target_seq_valid, -1) 
                feat_seq_train, target_seq_train = np.delete(feat_seq_train, -1, axis=0), np.delete(target_seq_train, -1) 
                feat_seq_valid, target_seq_valid = np.delete(feat_seq_valid, -1, axis=0), np.delete(target_seq_valid, -1) 
            else:
                feat_seq_train, target_seq_train = np.delete(feat_seq_train, -1, axis=1), np.delete(target_seq_train, -1, axis=1) 
                feat_seq_valid, target_seq_valid = np.delete(feat_seq_valid, -1, axis=1), np.delete(target_seq_valid, -1, axis=1) 
                
            
            train_input.append(feat_seq_train)
            train_target.append(target_seq_train)
            
            valid_input.append(feat_seq_valid)
            valid_target.append(target_seq_valid)
           
            
        return array(train_input, dtype='uint8'), array(train_target, dtype='uint8'), array(valid_input, dtype='uint8'), array(valid_target, dtype='uint8')



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
        
    
    traintext, validtext, num_classes, int_to_vocab, vocab_to_int = open_data(train_directory, valid_directory, num_files)# data_file = '/home/scheppacha/data/trainset.txt'
    
    train_feat, train_targ, valid_feat, valid_targ = create_sequences(traintext = traintext, validtext = validtext, seq_size = seq_size, next_character = next_character)
    
    
    train = get_data(train_feat, train_targ)# 
    valid = get_data(valid_feat, valid_targ)
    train_loader = torch.utils.data.DataLoader(train,batch_size,shuffle=True)#  shuffle ensures random choices of the sequences
    valid_loader = torch.utils.data.DataLoader(valid,batch_size,shuffle=True)
    
    return train_loader, valid_loader, num_classes