#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 19:52:20 2020

@author: amadeu
"""

#batch_size = 2

import csv
import gc
import DeepVirFinder_model as dvf
import data_preprocessing as dp
import torch
from DeepVirFinder_model import model, device, criterion, optimizer, num_motifs, num_classes
from data_preprocessing import train_loader, valid_loader, batch_size, num_batches_valid, num_batches_train, valid_y

import torch.nn as nn
import numpy as np 
#from pytorch_model_summary import summary



train_losses = []
valid_losses = []



def Train():
    
    running_loss = .0
    
    model.train() # dieses NN wurde oben als model definiert
    
    for idx, (inputs, labels) in enumerate(train_loader):
        #print('inputs')
        #print(inputs)
        #print(inputs.shape) # 32,100,4
        inputs = inputs.transpose(1, 2) # now 32,4,100
        #print('inputs_transposed')
        #print(inputs)
        #print(inputs.shape)
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        preds = model(inputs.float())

        print('preds')
        print(preds)
        print(preds.shape)

        labels = torch.max(labels, 1)[1]

        print('labels')
        print(labels)
        print(labels.shape)
        loss = criterion(preds,labels.long())# label is of shape [batchsize], which means, we have a true integer class for each batch;
        # preds is of shape [batchsize=32,num_classes=4], so we have probability for each batch, belonging to a certain class
        #loss.requires_grad = True

        loss.backward()
        optimizer.step()
        running_loss += loss
    
    train_loss = running_loss/len(train_loader)
    train_losses.append(train_loss.detach().cpu().numpy())

    print(f'train_loss {train_loss}')

def Valid():
   
    running_loss = .0
    
    model.eval()
        
    y_pred = np.zeros(1)
    y_true = np.zeros(1)

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(valid_loader):
            inputs = inputs.transpose(1, 2)
            inputs = inputs.to(device) 
            labels = labels.to(device)             
            optimizer.zero_grad()
            
            logits = model(inputs.float())
            labels = torch.max(labels, 1)[1]
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
    #return y_pred
    #with open('valid_loss.csv', 'w') as f:
    #    writer = csv.writer(f)
    #    writer.writerow(valid_losses)
    np.save('valid_loss', valid_losses)
    np.save('preds', y_pred)
    np.save('trues',y_true)
    #with open('preds.csv', 'w') as f: # logischerweise, macht er das für alle epochen, weshalb nur die letzte epoche abgespeichert wird (was uns nichts ausmacht, weil wir ja nur die prediction der letzten epoche haben wollen)
    #    writer = csv.writer(f)
    #    writer.writerow(y_pred)
    #with open('trues.csv', 'w') as f: # logischerweise, macht er das für alle epochen, weshalb nur die letzte epoche abgespeichert wird (was uns nichts ausmacht, weil wir ja nur die prediction der letzten epoche haben wollen)
    #    writer = csv.writer(f)
    #    writer.writerow(y_true)
    
