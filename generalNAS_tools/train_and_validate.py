#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 20:50:56 2021

@author: amadeu
"""


import torch
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from generalNAS_tools.utils import repackage_hidden, create_exp_dir, save_checkpoint 
from  generalNAS_tools import utils

import darts_tools.architect

import gc
#from torch.profiler import profile, record_function, ProfilerActivity

import logging
import sys
import torch.nn as nn
#log_format = '%(asctime)s %(message)s'
import torch.autograd.profiler as profiler


logging = logging.getLogger(__name__)
#logging.info("Hello logging!")



# train_queue, valid_queue =train_object, valid_object 
# train(train_object, valid_object, model, rhn, conv, criterion, optimizer, optimizer_a, None, args.unrolled, lr, epoch, args.num_steps, clip_params, args.report_freq, args.beta, args.one_clip, train_arch=False, pdarts=args.pdarts)


def train(train_queue, valid_queue, model, rhn, conv, criterion, optimizer, optimizer_a, architect, unrolled, lr, epoch, num_steps, clip_params, report_freq, beta, one_clip=True, train_arch=True, pdarts=True, task=None):
    objs = utils.AvgrageMeter()
    #top1 = utils.AvgrageMeter()
    #top5 = utils.AvgrageMeter()
    
    total_loss = 0
    start_time = time.time()
    #logging = logging.getLogger(__name__)
    #logging.info("Hello logging!")
    # train_queue = train_object
    scores = nn.Softmax()

    labels = []
    predictions = []
         
    for step, (input, target) in enumerate(train_queue): 
        
        if step > num_steps:
            break        
        
        input, target = input, target
       
        model.train()
        #input = input.transpose(1, 2).float()

        input = input.float().to(device)#.cuda()
        #target = torch.max(target, 1)[1]
        target = target.to(device)#.cuda(non_blocking=True)
        batch_size = input.size(0)
        
        hidden = model.init_hidden(batch_size) 
        
        if train_arch: 
            
            try:
                input_search, target_search = next(valid_queue_iter)
                input_search = input_search.float()

            except:
                valid_queue_iter = iter(valid_queue)
                input_search, target_search = next(valid_queue_iter)
                input_search = input_search.float()
                
            input_search = input_search.to(device)#.cuda()
            target_search = target_search.to(device)#.cuda(non_blocking=True)
            #target_search = torch.max(target_search, 1)[1]
            
            if pdarts:
                optimizer_a.zero_grad()
                
                # rnn_hs is output from RHN/LSTM layer before dropout and linear layer; dropped_rnn_hs is output from 
                # RHN/LSTM after LSTM and dropout but befor linear layer
                # he uses these tensor for loss regularization (penalize big differences between different batches)
                # as we don't have time dependecy between batches, we don't need this pard
                logits, hidden, rnn_hs, dropped_rnn_hs = model(input_search, hidden, return_h=True)
    
                loss_a = criterion(logits, target_search)
                loss_a.backward()
                nn.utils.clip_grad_norm_(model.arch_parameters(), clip_params[2]) 
                # Darts makes no clipping in the CNN, in the RHN they use clipping param of 0.25;
                # Pdarts uses clip param of 5 for CNN
                optimizer_a.step()
            else:
                hidden_valid = model.init_hidden(batch_size) 

                hidden_valid, grad_norm = architect.step( 
                                            hidden, input, target,
                                            hidden_valid, input_search, target_search,
                                            optimizer,
                                            unrolled)
                

        optimizer.zero_grad()
        
        hidden = model.init_hidden(batch_size)
        
        #with profiler.profile(with_stack=True, profile_memory=True) as prof:
        #with record_function("model_inference"):

        logits, hidden, rnn_hs, dropped_rnn_hs = model(input, hidden, return_h=True)
        
            
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        # print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))

        raw_loss = criterion(logits, target)
        loss = raw_loss
       
        # Temporal Activation Regularization (slowness)
        loss = loss + sum(beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        total_loss += raw_loss.data
        loss.backward()
      
        gc.collect()

        if one_clip == True:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_params[2])
        else:
            torch.nn.utils.clip_grad_norm_(conv, clip_params[0])
            torch.nn.utils.clip_grad_norm_(rhn, clip_params[1])
            
        optimizer.step()
        
        #prec1,prec2 = utils.accuracy(logits, target, topk=(1,2)) 
                
        objs.update(loss.data, batch_size)
        

        
        
        labels.append(target)
        if task == "next_character_prediction":
            predictions.append(scores(logits).detach().numpy())
        else:#if args.task == "TF_bindings"::
            predictions.append(logits.detach().numpy())
       
        if step % report_freq == 0 and step > 0:
        
            #logging.info('| step {:3d} | train obj {:5.2f} | '
            #    'train acc {:8.2f}'.format(step,
            #                               objs.avg, top1.avg))
            logging.info('| step {:3d} | train obj {:5.2f}'.format(step, objs.avg))
            
    if train_arch==True:
        logging.info('{} epoch done   '.format(epoch) + str(model.genotype()))

    return labels, predictions, objs.avg # top1.avg, objs.avg


def infer(valid_queue, model, criterion, batch_size, num_steps, report_freq, task=None):
    objs = utils.AvgrageMeter()
    #top1 = utils.AvgrageMeter()
    #top2 = utils.AvgrageMeter()

    model.eval()
    
    total_loss = 0
    labels = []
    predictions = []
    
    scores = nn.Softmax()


    for step, (input, target) in enumerate(valid_queue):
        
        if step > num_steps:
            break
        
        # input = input.transpose(1,2).float()
        input = input.to(device).float()
        batch_size = input.size(0)

        target = target.to(device)
        #target = torch.max(target, 1)[1]
        hidden = model.init_hidden(batch_size)#.to(device)  

        with torch.no_grad():
            logits, hidden = model(input, hidden)
            loss = criterion(logits, target)

        # prec1, prec5 = utils.accuracy(logits, target, topk=(1, 2))
        
        objs.update(loss.data, batch_size)
        labels.append(target)
        if task == "next_character_prediction":
            predictions.append(scores(logits).detach().numpy())
        else:#if args.task == "TF_bindings"::
            predictions.append(logits.detach().numpy())

        if step % report_freq == 0:
            #logging.info('| step {:3d} | val obj {:5.2f} | '
            #    'val acc {:8.2f}'.format(step,
            #                               objs.avg, top1.avg))
            logging.info('| step {:3d} | val obj {:5.2f}'.format(step, objs.avg))


    return labels, predictions, objs.avg # top1.avg, objs.avg