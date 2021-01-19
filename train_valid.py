#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 09:12:06 2021

@author: amadeu
"""

import csv
import gc
#import DeepVirFinder_model as dvf
import data_preprocessing as dp
import torch
#from DeepVirFinder_model import model, device, criterion, optimizer, num_motifs, num_classes
#from data_preprocessing import train_loader, valid_loader, batch_size, num_batches_valid, num_batches_train, valid_y

import torch.nn as nn
import numpy as np 
from sklearn.metrics import classification_report



train_losses = []
valid_losses = []
acc_train = []
acc_test = []


def Train(model, device, optimizer, criterion, train_loader, valid_loader, batch_size):
    
    running_loss = .0
    
    model.train() 
    
    y_true_train = np.empty((0,1), int)
    y_pred_train = np.empty((0,1), int)
    
    for idx, (inputs, labels) in enumerate(train_loader):
      
        inputs = inputs.transpose(1, 2) # now 2,4,10
     
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        inputs = inputs.type(torch.cuda.LongTensor)
      
        
        optimizer.zero_grad()

        logits = model(inputs)

        labels = torch.max(labels, 1)[1]

        loss = criterion(logits,labels.long())
        logits = torch.max(logits, 1)[1]

        labels, logits = labels.cpu().detach().numpy(), logits.cpu().detach().numpy()
         
        y_true_train = np.append(y_true_train, labels)
        y_pred_train = np.append(y_pred_train, logits)
        
        loss_value = loss.item()

        loss.backward()
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
    #with open('preds.csv', 'w') as f: # logischerweise, macht er das für alle epochen, weshalb nur die letzte epoche abgespeichert wird (was uns nichts ausmacht, weil wir ja nur die prediction der letzten epoche haben wollen)
    #    writer = csv.writer(f)
    #    writer.writerow(y_pred)
    #with open('trues.csv', 'w') as f: # logischerweise, macht er das für alle epochen, weshalb nur die letzte epoche abgespeichert wird (was uns nichts ausmacht, weil wir ja nur die prediction der letzten epoche haben wollen)
    #    writer = csv.writer(f)
    #    writer.writerow(y_true)