#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 20:58:07 2020

@author: amadeu
"""
#batch_size = 2
gradients_norm = 5

import csv
import gc
import data_preprocessing as dp
import torch

import torch.nn as nn
import numpy as np 
from sklearn.metrics import classification_report

train_losses = []
valid_losses = []
acc_train = []
acc_test = []


def Train(model, device, optimizer, criterion, train_loader, valid_loader, batch_size):
    
    running_loss = .0
    
    state_h, state_c = model.zero_state(batch_size)# zero_state wurde oben innerhalb von unserer NN modul definiert um states nach jeder epoche zu reseten
    state_h = state_h.to(device)
    state_c = state_c.to(device)
     
    y_true_train = np.empty((0,1), int)
    y_pred_train = np.empty((0,1), int)
    
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
        #print(logits)
        logits = torch.max(logits, 1)[1]

        labels, logits = labels.cpu().detach().numpy(), logits.cpu().detach().numpy()
         
        y_true_train = np.append(y_true_train, labels)
        y_pred_train = np.append(y_pred_train, logits)
            
        state_h = state_h.detach()
        state_c = state_c.detach()

        loss_value = loss.item()

        loss.backward()#retain_graph=True) -> we will need this, for training two branches in parallel

        optimizer.step()
        
        _ = torch.nn.utils.clip_grad_norm_(model.parameters(), gradients_norm) # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.

        optimizer.step()
            
        running_loss += loss
     
    train_loss = running_loss/len(train_loader)
    train_losses.append(train_loss.detach().cpu().numpy())
    precRec = classification_report(y_pred_train, y_true_train, output_dict=True)
    acc_train_epoch = precRec['accuracy']       
    acc_train.append(acc_train_epoch)
    
    print(f'train_loss {train_loss}')
    
    np.save('train_loss', train_losses)
    np.save('acc_train', acc_train)

    np.save('preds_train', y_pred_train)
    np.save('trues_train',y_true_train)

    
    
def Valid(model, device, optimizer, criterion, train_loader, valid_loader, batch_size):
   
    running_loss = .0
    
    model.eval()

    state_h, state_c = model.zero_state(batch_size) #reset all sates (input is always batchsize)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    
    y_pred = np.empty((0,1), int)
    y_true = np.empty((0,1), int)


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
        
        precRec = classification_report(y_pred, y_true, output_dict=True)
        acc_test_epoch = precRec['accuracy']
            
        acc_test.append(acc_test_epoch)

        valid_loss = running_loss/len(valid_loader)
        valid_losses.append(valid_loss.detach().cpu().numpy())
        print(f'valid_loss {valid_loss}')
        
    np.save('acc_test', acc_test)
    np.save('valid_loss', valid_losses)
    np.save('preds', y_pred)
    np.save('trues',y_true)
    
    
    
   
        
