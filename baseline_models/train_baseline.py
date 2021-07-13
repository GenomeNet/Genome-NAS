#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 12:33:30 2021

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

from  generalNAS_tools import utils


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import csv
import gc
#import DeepVirFinder_model as dvf
import generalNAS_tools.data_preprocessing_new as dp
import torch
#from DeepVirFinder_model import model, device, criterion, optimizer, num_motifs, num_classes
#from data_preprocessing import train_loader, valid_loader, batch_size, num_batches_valid, num_batches_train, valid_y

import torch.nn as nn
import numpy as np 
from sklearn.metrics import classification_report


parser = argparse.ArgumentParser("train_LSTM_based")
parser.add_argument('--data', type=str, default='/home/amadeu/anaconda3/envs/darts_env/cnn/data2/trainset.txt', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=2, help='batch size')
parser.add_argument('--seq_size', type=int, default=1000, help='sequence size') # 200 oder 1000
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--epochs', type=int, default=10, help='num of training epochs')
parser.add_argument('--num_classes', type=int, default=4, help='num of target classes')
parser.add_argument('--task', type=str, default='next_character_prediction', help='defines the task')#TF_bindings
parser.add_argument('--num_steps', type=int, default=2, help='number of iterations per epoch')
parser.add_argument('--valid_directory', type=str, default='/home/amadeu/Downloads/genomicData/validation', help='directory of validation data')
parser.add_argument('--train_directory', type=str, default='/home/amadeu/Downloads/genomicData/train', help='directory of training data')
parser.add_argument('--num_files', type=int, default=3, help='number of files for training data')
parser.add_argument('--next_character_prediction', type=bool, default=True, help='task of model')
parser.add_argument('--note', type=str, default='try', help='note for this run')

parser.add_argument('--train_input_directory', type=str, default='/home/amadeu/Desktop/GenomNet_MA/data/inputs_small.pkl', help='directory of validation data')
parser.add_argument('--train_target_directory', type=str, default='/home/amadeu/Desktop/GenomNet_MA/data/targets_small.pkl', help='directory of validation data')
parser.add_argument('--valid_input_directory', type=str, default='/home/amadeu/Desktop/GenomNet_MA/data/inputs_small_val.pkl', help='directory of validation data')
parser.add_argument('--valid_target_directory', type=str, default='/home/amadeu/Desktop/GenomNet_MA/data/targets_small_val.pkl', help='directory of validation data')


parser.add_argument('--num_motifs', type=int, default=100, help='number of channels') # 320
parser.add_argument('--model', type=str, default='DanQ', help='path to save the model')
parser.add_argument('--save', type=str,  default='EXP',
                    help='path to save the final model')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--report_freq', type=int, default=5, help='validation report frequency')
args = parser.parse_args()


args.save = '{}search-{}-{}'.format(args.save, args.note, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))


def main():
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  # criterion = nn.BCELoss()
  
  if (args.task == "next_character_prediction"):
      import generalNAS_tools.data_preprocessing_new as dp
      
      train_queue, valid_queue, num_classes = dp.data_preprocessing(train_directory = args.train_directory, valid_directory = args.valid_directory, num_files=args.num_files,
                seq_size = args.seq_size, batch_size=args.batch_size, next_character=args.next_character_prediction)

      if (args.model == "DanQ"):
          import baseline_models.models.DanQ_model as model

          model = model.NN_class(args.num_classes, args.num_motifs, args.batch_size, args.seq_size, args.task).to(device)
          criterion = nn.CrossEntropyLoss().to(device)
          optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)
          
      if (args.model == "DeepSEA"):
          import baseline_models.models.DeepSea as model
          model = model.NN_class(args.num_classes, args.batch_size, args.seq_size, args.task).to(device)
          criterion = nn.CrossEntropyLoss().to(device)
          optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=1e-6, momentum=0.9)
        

      
      
  if (args.task == "TF_bindings"):
      
      import generalNAS_tools.data_preprocessing_TF as dp
        
      train_queue, valid_queue = dp.data_preprocessing(args.train_input_directory, args.valid_input_directory, args.train_target_directory, args.valid_target_directory, args.batch_size)
      
      if (args.model == "DanQ"):
          
          import baseline_models.models.DanQ_model as model

          model = model.NN_class(args.num_classes, args.num_motifs, args.batch_size, args.seq_size, args.task).to(device)
          criterion = nn.BCELoss().to(device)
          optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)

          
      if (args.model == "DeepSEA"):
          
          import baseline_models.models.DeepSea as model

          model = models.DanQ_model.NN_class(args.num_classes, args.num_motifs, args.batch_size, args.seq_size).to(device)
          criterion = nn.BCELoss().to(device)
          optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=1e-6, momentum=0.9)

  
  # '/home/amadeu/anaconda3/envs/darts_env/cnn/data2/trainset.txt'
  # '/home/ascheppa/miniconda2/envs/darts/cnn/data2/trainset.txt'
  # train_queue, valid_queue, num_classes = dp.data_preprocessing(data_file =args.data, 
  #        seq_size = args.seq_size, representation = 'onehot', model_type = 'CNN', batch_size=args.batch_size)
  
  train_losses = []
  valid_losses = []
  acc_train = []
  acc_valid = []

  for epoch in range(args.epochs):
      
      
      train_start = time.strftime("%Y%m%d-%H%M")

      
      train_loss, acc_train_epoch = Train(model, train_queue, optimizer, criterion, device, args.num_steps, args.report_freq)
      
      train_losses.append(train_loss)
      acc_train.append(acc_train_epoch)
      # np.save('train_loss', train_losses)
      # np.save('acc_train', acc_train)
      
      if epoch % args.report_freq == 0:
          
          valid_loss, acc_test_epoch = Valid(model, train_queue, valid_queue, optimizer, criterion, device, args.num_steps, args.report_freq)
          
          valid_losses.append(valid_loss)
          acc_valid.append(acc_test_epoch)
          # np.save('acc_test', acc_test)
          # np.save('valid_loss', valid_losses)
        
      trainloss_file = '{}-train_loss-{}'.format(args.save, train_start)
      np.save(trainloss_file, train_losses)
      
      acctrain_file = '{}-acc_train-{}'.format(args.save, train_start) 
      np.save(acctrain_file, acc_train)
      
      validloss_file = '{}-valid_loss-{}'.format(args.save, train_start)
      np.save(validloss_file, valid_losses)
      
      accvalid_file = '{}-acc_valid-{}'.format(args.save, train_start) 
      np.save(acctrain_file, acc_valid)

      
    

# train_loader, num_steps = train_queue, 2

def Train(model, train_loader, optimizer, criterion, device, num_steps, report_freq):
    
    #running_loss = .0
    
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    
    total_loss = 0
    start_time = time.time()
    
    #state_h, state_c = model.zero_state(args.batch_size)# zero_state wurde oben innerhalb von unserer NN modul definiert um states nach jeder epoche zu reseten
    #state_h = state_h.to(device)
    #state_c = state_c.to(device)
        
    #y_true_train = np.empty((0,1), int)
    #y_pred_train = np.empty((0,1), int)
    
    for idx, (inputs, labels) in enumerate(train_loader):
        
        if idx > num_steps:
            break
        
        input, label = inputs, labels
       
        model.train()
        #input = input.transpose(1, 2).float()

        input = input.float().to(device)#.cuda()
        
       
        batch_size = input.size(0)
        
        
        #state_h, state_c = model.zero_state(args.batch_size)# zero_state wurde oben innerhalb von unserer NN modul definiert um states nach jeder epoche zu reseten
        #state_h = state_h.to(device)
        #state_c = state_c.to(device)
        #logits, (state_h, state_c) = model(input)
        optimizer.zero_grad()
        
         
        if args.task == "TF_bindings":
            logits = model(input.float()) #, (state_h, state_c))

        # label = torch.max(label, 1)[1]

            loss = criterion(logits, label)
        else:
            label = torch.max(label, 1)[1]
            label = label.to(device)#.cuda(non_blocking=True)
            logits = model(input.float()) #, (state_h, state_c))

            # label = torch.max(label, 1)[1]

            loss = criterion(logits, label.long())

        
        #logits = torch.max(logits, 1)[1]

        #labels, logits = label.cpu().detach().numpy(), logits.cpu().detach().numpy()
         
        #y_true_train = np.append(y_true_train, labels)
        #y_pred_train = np.append(y_pred_train, logits)
        
        #state_h = state_h.detach()
        #state_c = state_c.detach()
        
        #loss_value = loss.item()

        loss.backward()
        optimizer.step()
        #running_loss += loss
        
        prec1, prec2 = utils.accuracy(logits, label, topk=(1,2)) 
                
        objs.update(loss.data, batch_size)
       
        top1.update(prec1.data, batch_size) 
        
        if idx % report_freq == 0 and idx > 0:
        
            logging.info('| step {:3d} | train obj {:5.2f} | '
                'train acc {:8.2f}'.format(idx,
                                           objs.avg, top1.avg))
    
    #train_loss = running_loss/len(train_loader)
    #train_loss = train_loss.detach().cpu().numpy()
    #train_losses.append(train_loss.detach().cpu().numpy())
    #precRec = classification_report(y_pred_train, y_true_train, output_dict=True)
    #acc_train_epoch = precRec['accuracy']       
    #acc_train.append(acc_train_epoch)
    
    #print(f'acc_training {acc_train_epoch}')
    
    #np.save('train_loss', train_losses)
    #np.save('acc_train', acc_train)

    #np.save('preds_train', y_pred_train)
    #np.save('trues_train',y_true_train)
    
    return top1.avg, objs.avg # train_loss, acc_train_epoch
    

def Valid(model, train_loader, valid_loader, optimizer, criterion, device, num_steps, report_freq):
   
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    
    total_loss = 0
    start_time = time.time()
    
    #state_h, state_c = model.zero_state(args.batch_size)# zero_state wurde oben innerhalb von unserer NN modul definiert um states nach jeder epoche zu reseten
    #state_h = state_h.to(device)
    #state_c = state_c.to(device)
    
    model.eval()

    with torch.no_grad():
        
        for idx, (inputs, labels) in enumerate(valid_loader):
            
            
            if idx > num_steps:
                break
        
            input, label = inputs, labels
           
            #input = input.transpose(1, 2).float()
    
            input = input.float().to(device)#.cuda()
            
            label = torch.max(label, 1)[1]
            label = label.to(device)#.cuda(non_blocking=True)
            batch_size = input.size(0)
            
            #state_h, state_c = model.zero_state(args.batch_size)# zero_state wurde oben innerhalb von unserer NN modul definiert um states nach jeder epoche zu reseten
            #state_h = state_h.to(device)
            #state_c = state_c.to(device)
            #logits, (state_h, state_c) = model(input)
    
            logits = model(input.float())#, (state_h, state_c))
    
            # label = torch.max(label, 1)[1]
    
            loss = criterion(logits, label.long())
            
           
            #running_loss += loss

            # logits = torch.max(logits, 1)[1]
            
            prec1, prec2 = utils.accuracy(logits, label, topk=(1,2)) 
                
            objs.update(loss.data, batch_size)
       
            top1.update(prec1.data, batch_size) 
        
            if idx % report_freq == 0 and idx > 0:
        
                logging.info('| step {:3d} | valid obj {:5.2f} | '
                             'train acc {:8.2f}'.format(idx,
                                           objs.avg, top1.avg))
    
    
            
            
        #precRec = classification_report(y_pred, y_true, output_dict=True)
        #acc_test_epoch = precRec['accuracy']
            
        #valid_loss = running_loss/len(valid_loader)
        #valid_loss = valid_loss.detach().cpu().numpy()
        
        #print(f'valid_accuracy {acc_test_epoch}')
        
    #with open('preds.csv', 'w') as f: 
    #    writer = csv.writer(f)
    #    writer.writerow(y_pred)
    #with open('trues.csv', 'w') as f: 
    #    writer = csv.writer(f)
    #    writer.writerow(y_true)
    return top1.avg, objs.avg


if __name__ == '__main__':
  main() 