#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 20:58:07 2020

@author: amadeu
"""
batch_size = 20
gradients_norm = 5

import csv
import gc
import CNN_LSTM_comb as cl
import data_preprocessing as dp
import torch
from CNN_LSTM_comb import model, device, criterion, optimizer
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
        
        inputs = torch.reshape(inputs, (batch_size, 1, 5))

        inputs = torch.tensor(inputs).to(device)
        labels = torch.tensor(labels).to(device)
        
        
        logits, (state_h, state_c) = model(inputs.float(), (state_h, state_c))
        logits = torch.reshape(logits, (32,4))
        #print('logits')
        #print(logits)
        #print(logits.shape)
        
        #print('labels')
        #print(labels)
        #print(labels.shape)
      
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
    # bis hierhin passt
    
    y_pred = np.zeros(((np.prod(valid_y.shape) // (batch_size)),batch_size), dtype = str)# ich habe 500.000 valid werte, welche sich auf 32er batches verteilen mit jeweils 5er sequenzen und das für 100 epochs

    # epochenanzahl ist wurscht, weil er in main() file diese jedes mal von neuem ausführt, also bleibt nur letzte epoche
    y_true = np.zeros(((np.prod(valid_y.shape) // (batch_size)),batch_size), dtype = str)# haben 32 (batch_size) spalten und 3124 zeilen (num_valid_batches)

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(valid_loader):
            inputs = torch.reshape(inputs, (batch_size, 1, 5))
            
            inputs = inputs.to(device) 
            labels = labels.to(device) 
            print('labels')
            print(labels)
            optimizer.zero_grad()
            logits, (state_h, state_c) = model(inputs.float(), (state_h, state_c))
            logits = torch.reshape(logits, (32,4))
            
            _, predicted = torch.max(logits, 1)    
           
            for i in range(valid_loader.batch_size):
               print(predicted[i])
               predicted0 = predicted[i].item()
               y_pred[idx,i] = predicted0
               
               label0 =  labels[i].item()
               y_true[idx,i] = label0
               
            loss = criterion(logits, labels.long())
            running_loss += loss
            
        valid_loss = running_loss/len(valid_loader)
        valid_losses.append(valid_loss.detach().cpu().numpy())
        print(f'valid_loss {valid_loss}')
    #return y_pred
    with open('valid_loss.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(valid_losses)
    with open('preds.csv', 'w') as f: # logischerweise, macht er das für alle epochen, weshalb nur die letzte epoche abgespeichert wird (was uns nichts ausmacht, weil wir ja nur die prediction der letzten epoche haben wollen)
        writer = csv.writer(f)
        writer.writerow(y_pred)
    with open('trues.csv', 'w') as f: # logischerweise, macht er das für alle epochen, weshalb nur die letzte epoche abgespeichert wird (was uns nichts ausmacht, weil wir ja nur die prediction der letzten epoche haben wollen)
        writer = csv.writer(f)
        writer.writerow(y_true)
    
    
   
        
