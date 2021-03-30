#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 11:48:33 2021

@author: amadeu
"""

import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import data_preprocessing as dp

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
import models.DanQ_model


parser = argparse.ArgumentParser("train_LSTM_based")
parser.add_argument('--data', type=str, default='/home/amadeu/anaconda3/envs/darts_env/cnn/data2/trainset.txt', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=2, help='batch size')
parser.add_argument('--seq_size', type=int, default=10, help='sequence size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--epochs', type=int, default=10, help='num of training epochs')
parser.add_argument('--num_classes', type=int, default=4, help='num of target classes')
parser.add_argument('--num_motifs', type=int, default=100, help='number of channels')
parser.add_argument('--model', type=str, default='DanQ', help='path to save the model')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()



def main():
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  criterion = nn.CrossEntropyLoss().to(device)
  
  if (args.model == "DanQ"):
            model = models.DanQ_model.NN_class(args.num_classes, args.num_motifs, args.batch_size).to(device)
            
            
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
  
  # '/home/amadeu/anaconda3/envs/darts_env/cnn/data2/trainset.txt'
  # '/home/ascheppa/miniconda2/envs/darts/cnn/data2/trainset.txt'
  train_queue, valid_queue, num_classes = dp.data_preprocessing(data_file =args.data, 
          seq_size = args.seq_size, representation = 'onehot', model_type = 'CNN', batch_size=args.batch_size)
  
  train_losses = []
  valid_losses = []
  acc_train = []
  acc_test = []

  for epoch in range(args.epochs):
      
      train_loss, acc_train_epoch = Train(model, train_queue, optimizer, criterion, device)
      
      train_losses.append(train_loss)
      acc_train.append(acc_train_epoch)
      np.save('train_loss', train_losses)
      np.save('acc_train', acc_train)
      
      valid_loss, acc_test_epoch = Valid(model, train_queue, valid_queue, optimizer, criterion, device)
      
      valid_losses.append(valid_loss)
      acc_test.append(acc_test_epoch)
      np.save('acc_test', acc_test)
      np.save('valid_loss', valid_losses)

      
    

# train_loader=train_queue 

def Train(model, train_loader, optimizer, criterion, device):
    
    running_loss = .0
    
    state_h, state_c = model.zero_state(args.batch_size)# zero_state wurde oben innerhalb von unserer NN modul definiert um states nach jeder epoche zu reseten
    state_h = state_h.to(device)
    state_c = state_c.to(device)
        
    y_true_train = np.empty((0,1), int)
    y_pred_train = np.empty((0,1), int)
    
    for idx, (inputs, labels) in enumerate(train_loader):
        
        #if idx > 1000:
        #    break
        
        model.train() 
 
        inputs = inputs.transpose(1, 2)
       
        inputs = torch.tensor(inputs).to(device)
        
        labels = torch.tensor(labels).to(device)
        
        logits, (state_h, state_c) = model(inputs.float(), (state_h, state_c))

        labels = torch.max(labels, 1)[1]

        loss = criterion(logits,labels.long())
        logits = torch.max(logits, 1)[1]

        labels, logits = labels.cpu().detach().numpy(), logits.cpu().detach().numpy()
         
        y_true_train = np.append(y_true_train, labels)
        y_pred_train = np.append(y_pred_train, logits)
        
        state_h = state_h.detach()
        state_c = state_c.detach()
        
        loss_value = loss.item()

        loss.backward()
        optimizer.step()
        running_loss += loss
    
    train_loss = running_loss/len(train_loader)
    train_loss = train_loss.detach().cpu().numpy()
    #train_losses.append(train_loss.detach().cpu().numpy())
    precRec = classification_report(y_pred_train, y_true_train, output_dict=True)
    acc_train_epoch = precRec['accuracy']       
    #acc_train.append(acc_train_epoch)
    
    print(f'acc_training {acc_train_epoch}')
    
    #np.save('train_loss', train_losses)
    #np.save('acc_train', acc_train)

    np.save('preds_train', y_pred_train)
    np.save('trues_train',y_true_train)
    
    return train_loss, acc_train_epoch
    

def Valid(model, train_loader, valid_loader, optimizer, criterion, device):
   
    running_loss = .0
    
    model.eval()
    
    state_h, state_c = model.zero_state(args.batch_size) #reset all sates (input is always batchsize)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
        
    y_pred = np.zeros(1)
    y_true = np.zeros(1)

    with torch.no_grad():
        
        for idx, (inputs, labels) in enumerate(valid_loader):
            inputs = inputs.transpose(1, 2)
            inputs = inputs.to(device) 
            labels = labels.to(device)             
            optimizer.zero_grad()
            
            logits, (state_h, state_c) = model(inputs.float(), (state_h, state_c))
            labels = torch.max(labels, 1)[1]
            loss = criterion(logits, labels.long())
            running_loss += loss

            logits = torch.max(logits, 1)[1]
            
            labels, logits = labels.cpu().detach().numpy(), logits.cpu().detach().numpy()
          
            y_true = np.append(y_true, labels)
            y_pred = np.append(y_pred, logits)
            
            
        precRec = classification_report(y_pred, y_true, output_dict=True)
        acc_test_epoch = precRec['accuracy']
            
        valid_loss = running_loss/len(valid_loader)
        valid_loss = valid_loss.detach().cpu().numpy()
        
        print(f'valid_accuracy {acc_test_epoch}')
        
   
    np.save('preds', y_pred)
    np.save('trues',y_true)
    #with open('preds.csv', 'w') as f: 
    #    writer = csv.writer(f)
    #    writer.writerow(y_pred)
    #with open('trues.csv', 'w') as f: 
    #    writer = csv.writer(f)
    #    writer.writerow(y_true)
    return valid_loss, acc_test_epoch


if __name__ == '__main__':
  main() 
