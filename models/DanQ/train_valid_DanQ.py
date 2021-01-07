#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 20:58:07 2020

@author: amadeu
"""
batch_size = 32
gradients_norm = 5

import csv
import gc
import DanQ_model as dq
import data_preprocessing as dp
import torch
from DanQ_model import model, device, criterion, optimizer
from data_preprocessing import train_loader, valid_loader, batch_size, num_batches_valid, num_batches_train, valid_y

import torch.nn as nn
import numpy as np 



train_losses = []
valid_losses = []

def Train():
    
    running_loss = .0
    
    state_h, state_c = model.zero_state(batch_size)# zero_state wurde oben innerhalb von unserer NN modul definiert um states nach jeder epoche zu reseten
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    
    #model.train() 
    
    for idx, (inputs, labels) in enumerate(train_loader):
        
        model.train()
        optimizer.zero_grad()
        
        #inputs = torch.reshape(inputs, (batch_size, 1, 5))
        inputs = inputs.transpose(1, 2)

        inputs = torch.tensor(inputs).to(device)
        labels = torch.tensor(labels).to(device)
        
        logits, (state_h, state_c) = model(inputs.float(), (state_h, state_c))

       
        # DeepVirFinder: labels hat shape 2,4 ; logits shape 2,6,4 
        # offizielle pytorch dokumentation: labels shape 2(wahre klasse als int), logits shape 2,4 (wahrscheinlichkeiten für jede klasse)
        #print(labels)
        labels = torch.max(labels, 1)[1]
        
        loss = criterion(logits, labels.long())

        state_h = state_h.detach()
        state_c = state_c.detach()

        loss_value = loss.item()

        loss.backward(retain_graph=True)

        optimizer.step()
        
        _ = torch.nn.utils.clip_grad_norm_(model.parameters(), gradients_norm) # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.

            
        optimizer.step()
            
        running_loss += loss
        
    train_loss = running_loss/len(train_loader)
    train_losses.append(train_loss.detach().cpu().numpy())
    
    print(f'train_loss {train_loss}')
    
def Valid():
   
    running_loss = .0
    
    model.eval()

    state_h, state_c = model.zero_state(batch_size) #reset all sates (input is always batchsize)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    
    y_pred = np.zeros(1)
    y_true = np.zeros(1)


    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(valid_loader):
            inputs = inputs.transpose(1, 2)
            inputs = torch.tensor(inputs).to(device)
            labels = torch.tensor(labels).to(device)
            optimizer.zero_grad()
            
            logits, (state_h, state_c) = model(inputs.float(), (state_h, state_c))
            labels = torch.max(labels, 1)[1]

            #logits = torch.reshape(logits, (32,4))
            #_, predicted = torch.max(logits, 1)    
      
            loss = criterion(logits, labels.long())
            running_loss += loss
            
            logits = torch.max(logits, 1)[1]
            labels, logits = labels.cpu().detach().numpy(), logits.cpu().detach().numpy()
          
            y_true = np.append(y_true, labels)
            y_pred = np.append(y_pred, logits)
        
        y_true, y_pred = np.delete(y_true, (0), axis = 0), np.delete(y_pred, (0), axis = 0)
        y_true, y_pred = y_true.astype(np.int), y_pred.astype(np.int)

        valid_loss = running_loss/len(valid_loader)
        valid_losses.append(valid_loss.detach().cpu().numpy())
        print(f'valid_loss {valid_loss}')
    
    np.save('valid_loss', valid_losses)
    np.save('preds', y_pred)
    np.save('trues',y_true)
    
    
    
   
        
