#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 09:28:56 2021

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
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--seq_size', type=int, default=1000, help='sequence size') # 200 oder 1000
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--epochs', type=int, default=10, help='num of training epochs')
parser.add_argument('--num_classes', type=int, default=919, help='num of target classes')
parser.add_argument('--task', type=str, default='TF_bindings', help='defines the task')#TF_bindings
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
parser.add_argument('--model_path', type=str,  default='./danQ_net.pth',
                    help='path to save the trained model')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--report_freq', type=int, default=5, help='validation report frequency')
args = parser.parse_args()


args.save = '{}search-{}-{}'.format(args.save, args.note, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))


def main():
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  if (args.task == "next_character_prediction"):
      import generalNAS_tools.data_preprocessing_new as dp
      
      train_queue, valid_queue, num_classes = dp.data_preprocessing(train_directory = args.train_directory, valid_directory = args.valid_directory, num_files=args.num_files,
                seq_size = args.seq_size, batch_size=args.batch_size, next_character=args.next_character_prediction)

      if (args.model == "DanQ"):
          model = torch.load(args.model_path)
          criterion = nn.CrossEntropyLoss().to(device)
          optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)
          

          
      if (args.model == "DeepSEA"):
          model = torch.load(args.model_path)
          criterion = nn.CrossEntropyLoss().to(device)
          optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=1e-6, momentum=0.9)
        
      
  if (args.task == "TF_bindings"):
      
      import generalNAS_tools.data_preprocessing_TF as dp
        
      train_queue, valid_queue = dp.data_preprocessing(args.train_input_directory, args.valid_input_directory, args.train_target_directory, args.valid_target_directory, args.batch_size)
      
      if (args.model == "DanQ"):
          
          model = torch.load(args.model_path)
          criterion = nn.BCELoss().to(device)
          optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)

          
      if (args.model == "DeepSEA"):
          
          model = torch.load(args.model_path)
          criterion = nn.BCELoss().to(device)
          optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=1e-6, momentum=0.9)
  
  # '/home/amadeu/anaconda3/envs/darts_env/cnn/data2/trainset.txt'
  # '/home/ascheppa/miniconda2/envs/darts/cnn/data2/trainset.txt'
  # train_queue, valid_queue, num_classes = dp.data_preprocessing(data_file =args.data, 
  #        seq_size = args.seq_size, representation = 'onehot', model_type = 'CNN', batch_size=args.batch_size)
  
  test_losses = []
  all_labels_test = []
  all_predictions_test = []

  train_start = time.strftime("%Y%m%d-%H%M")
  
  for epoch in range(args.epochs):
      
      labels, predictions, test_loss = Valid(model, train_queue, valid_queue, optimizer, criterion, device, args.num_steps, args.report_freq)
          
      test_losses.append(test_loss)
      all_labels_test.append(labels)
      all_predictions_test.append(predictions)
      
      
      
  testloss_file = '{}-test_loss-{}'.format(args.save, train_start)
  np.save(testloss_file, test_losses)
  labels_test_file = '{}-labels_test-{}'.format(args.save, train_start)
  np.save(labels_test_file, all_labels_test)
  predictions_test_file = '{}-predictions_test-{}'.format(args.save, train_start)
  np.save(predictions_test_file, all_predictions_test)
      



      

def Valid(model, train_loader, valid_loader, optimizer, criterion, device, num_steps, report_freq):
   
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    
    total_loss = 0
    start_time = time.time()

    
    model.eval()
    
    labels = []
    predictions = []

    with torch.no_grad():
        
        for idx, (inputs, labels) in enumerate(valid_loader):
            
            
            if idx > num_steps:
                break
        
            input, label = inputs, labels
               
            input = input.float().to(device)#.cuda()
            
            label = torch.max(label, 1)[1]
            label = label.to(device)#.cuda(non_blocking=True)
            batch_size = input.size(0)
            
            if args.task == "TF_bindings":
                logits = model(input.float()) #, (state_h, state_c))

                loss = criterion(logits, label)
            else:
                label = torch.max(label, 1)[1]
                label = label.to(device)#.cuda(non_blocking=True)
                logits = model(input.float()) #, (state_h, state_c))

                loss = criterion(logits, label.long())
            
            labels.append(label)
        
            predictions.append(logits)
                            
            objs.update(loss.data, batch_size)

    return labels, predictions, objs.avg #top1.avg, objs.avg


if __name__ == '__main__':
  main() 