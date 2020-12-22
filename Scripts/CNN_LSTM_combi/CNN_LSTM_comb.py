#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 14:06:32 2020

@author: amadeu
"""

import torch.nn as nn
import torch.nn.functional as F

import torch
#from torch.utils.data import Dataset,DataLoader
import data_preprocessing as dp


text, n_vocab, int_to_vocab, vocab_to_int = dp.open_data('/home/amadeu/Desktop/genom_Models/genom_venv/data/trainset.txt')#'/home/scheppacha/data/trainset.txt')

seq_size=5
lstm_size = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32

    
class CNN_LSTM_Module(nn.Module):
    def __init__(self, n_vocab, seq_size, lstm_size):
        super(CNN_LSTM_Module, self).__init__()
        self.seq_size = seq_size
        self.lstm_size = lstm_size
        self.conv1d = nn.Conv1d(1,4,kernel_size=2)# habe 1 input und output channels/feature maps mit 2 neuronen (weil input sind 3 neuronen, wo wir mit 2er kernels darüber laufen)
        self.batchnorm = nn.BatchNorm1d(4)
        self.relu = nn.ReLU(inplace=True)
        self.batchnorm = nn.BatchNorm1d(4)
        self.maxpool = nn.MaxPool1d(kernel_size = 2, stride = 1)
        self.fc1 = nn.Linear(4*3,6)
        self.lstm = nn.LSTM(6, #
                            lstm_size,
                            batch_first=True)
        self.dense = nn.Linear(lstm_size, n_vocab)
        
    def forward(self, x, prev_state):
        
        x = self.conv1d(x)
        x = self.batchnorm(x)
        #
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = x.view(-1,4*3)# in order to flatten, the 4 depth channel and 3 neurons
        #
        x = self.fc1(x)
       
        x = self.relu(x)
        x = torch.reshape(x, (32, 1, 6))# because LSTM always receives 3 dimension input: first is the sequences itself (batches), 
        # second are the instances of the minibatch (normally should be the num_steps) and the third is
        # the elements of the input (neurons): can try (2,6,1) as well
        
        output, state = self.lstm(x, prev_state)
        #print('output')
        #print(output)
        #print(output.shape)
        logits = self.dense(output)
        return logits, state
    # Because we need to reset states at the beginning of every epoch
    # (set all states to zero)
    def zero_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm_size),
                torch.zeros(1, batch_size, self.lstm_size))
    

model = CNN_LSTM_Module(n_vocab, seq_size, lstm_size)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


