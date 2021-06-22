#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 10:26:43 2021

@author: amadeu
"""


import os
import sys
import time
import glob
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import copy
import gc
import GenomNet_MA.darts_tools.model_search as one_shot_model
from GenomNet_MA.generalNAS_tools.genotypes import PRIMITIVES_cnn, PRIMITIVES_rnn, rnn_steps, CONCAT, Genotype
# from genotypes_rnn import PRIMITIVES, STEPS, CONCAT, total_genotype

import GenomNet_MA.generalNAS_tools.data_preprocessing_new as dp
# import general_tools.data_preprocessing_new

# sys.path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from GenomNet_MA.generalNAS_tools.utils import repackage_hidden, create_exp_dir, save_checkpoint 
import GenomNet_MA.generalNAS_tools.utils
from GenomNet_MA.generalNAS_tools.train_and_validate import train, infer

from GenomNet_MA.generalNAS_tools.utils import repackage_hidden, create_exp_dir, save_checkpoint 
from GenomNet_MA.generalNAS_tools import utils

from GenomNet_MA.darts_tools.final_stage_run import final_stage_genotype
from GenomNet_MA.darts_tools.auxiliary_functions import *
from GenomNet_MA.darts_tools.discard_operations import discard_cnn_ops, discard_rhn_ops

from GenomNet_MA.darts_tools.genotype_parser import parse_genotype




parser = argparse.ArgumentParser("cifar")
parser.add_argument('--single_gpu', default=True, action='store_false', 
                    help='use single GPU')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')

parser.add_argument('--cnn_lr', type=float, default=0.025, help='learning rate for CNN part')
parser.add_argument('--cnn_weight_decay', type=float, default=3e-4, help='weight decay for CNN part')
parser.add_argument('--rhn_lr', type=float, default=2, help='learning rate for RHN part')
parser.add_argument('--rhn_weight_decay', type=float, default=5e-7, help='weight decay for RHN part')
parser.add_argument('--validation', type=bool, default=True)
parser.add_argument('--report_validation', type=int, default=2, help='validation epochs') 
parser.add_argument('--workers', type=int, default=2, help='number of workers to load dataset')
parser.add_argument('--num_steps', type=int, default=2, help='number of iterations per epoch')
parser.add_argument('--train_directory', type=str, default='/home/amadeu/Downloads/genomicData/train', help='directory of training data')
parser.add_argument('--valid_directory', type=str, default='/home/amadeu/Downloads/genomicData/validation', help='directory of validation data')
parser.add_argument('--num_files', type=int, default=3, help='number of files for data')
parser.add_argument('--next_character_prediction', type=bool, default=True, help='task of model')
parser.add_argument('--one_clip', type=bool, default=True)
parser.add_argument('--clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--conv_clip', type=float, default=5, help='gradient clipping of convs')
parser.add_argument('--rhn_clip', type=float, default=0.25, help='gradient clipping of lstms')
parser.add_argument('--seq_len', type=int, default=200, help='sequence length')
parser.add_argument('--num_classes', type=int, default=4, help='num of output classes') # args.C, args.num_classes, args.layers, args.steps=4, args.multiplier=4, args.stem_multiplier=3
parser.add_argument('--steps', type=int, default=4, help='total number of Nodes')
parser.add_argument('--multiplier', type=int, default=4, help='multiplier')
parser.add_argument('--stem_multiplier', type=int, default=3, help='stem multiplier')
parser.add_argument('--batch_size', type=int, default=2, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--dropouth', type=float, default=0.25,
                    help='dropout for hidden nodes in rnn layers (0 = no dropout)')
parser.add_argument('--dropoutx', type=float, default=0.75,
                    help='dropout for input nodes in rnn layers (0 = no dropout)')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--report_freq', type=float, default=1000, help='report frequency')
parser.add_argument('--epochs', type=int, default=15, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=3, help='total number of layers')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
#parser.add_argument('--save', type=str, default='/tmp/checkpoints/', help='experiment path')
parser.add_argument('--beta', type=float, default=1e-3,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--save', type=str,  default='EXP',help='path to save the model')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--tmp_data_dir', type=str, default='/tmp/cache/', help='temp data dir')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--dropout_rate', action='append', default=[], help='dropout rate of skip connect')
parser.add_argument('--add_width', action='append', default=['0'], help='add channels')
parser.add_argument('--add_layers', action='append', default=['0'], help='add layers')
parser.add_argument('--cifar100', action='store_true', default=False, help='search with cifar100 dataset')
#parser.add_argument('--search', type=bool, default=True, help='if we search or evaluate architecture')
args = parser.parse_args()

args.save = '{}search-{}-{}'.format(args.save, args.note, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

    
def main():
 
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
  
    logging.info("args = %s", args)
   
    
    train_object, valid_object, num_classes = dp.data_preprocessing(train_directory = args.train_directory, valid_directory = args.valid_directory, num_files=args.num_files,
        seq_size = args.seq_len, batch_size=args.batch_size, next_character=args.next_character_prediction)

    _, valid_data, num_classes = dp.data_preprocessing(train_directory = args.train_directory, valid_directory = args.valid_directory, num_files=args.num_files,
        seq_size = args.seq_len, batch_size=args.batch_size, next_character=args.next_character_prediction)
 
    # build Network
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    # get switches for normal_cell and reduce_cell for CNN part
    switches_cnn = [] 
    for i in range(14):
        switches_cnn.append([True for j in range(len(PRIMITIVES_cnn))])
        
    switches_normal_cnn = copy.deepcopy(switches_cnn)
    switches_reduce_cnn = copy.deepcopy(switches_cnn)
    
    # get switches for RNN part
    switches_rnn = [] 
    for i in range(36):
        switches_rnn.append([True for j in range(len(PRIMITIVES_rnn))]) 
    switches_rnn = copy.deepcopy(switches_rnn)
    
    ## CNN ##
    num_to_keep = [8, 7, 6, 5, 4, 3, 1] # num_to_keep = [5, 3, 1]
    num_to_drop = [1, 1, 1, 1, 1, 1, 2] # num_to_drop = [3, 2, 2]
    
    discard_cnn_edges = [0, 0, 0, 0, 1, 2, 0]
    keep_cnn_edges = [14, 14, 14, 14, 13, 11, 11]
    disc_cnn_idxs = np.nonzero(discard_cnn_edges)

    # sp=0: 14,9 (trainiert) -> 14,8 (danach discarded)
    # sp=1: 14,8 -> 14,7
    # sp=2: 14,7 -> 14,6
    # sp=3: 14,6 -> 14,5
    # sp=4: 14,5 -> 14,4 -> 13,4
    # sp=5 (letzter sp wo discarded wird): 13,4 -> 13,3 -> 11,3
    # sp=6 (wird nur noch in final architecture umgewandelt): 11,3 -> 11,1 
    
    ## RHN ##
    num_to_keep_rnn = [4, 3, 2, 2, 2, 2, 1]
    num_to_drop_rnn = [1, 1, 1, 0, 0, 0, 1]
    disc_rhn_ops = np.nonzero(num_to_drop_rnn)
    
    discard_rhn_edges = [0, 1, 2, 3, 4, 5, 0]
    keep_rhn_edges = [36, 35, 33, 30, 26, 21, 21]
    disc_rhn_idxs = np.nonzero(discard_rhn_edges)
    # sp=0: 36,5 (trainiert) -> 35,4 (danach discarded)
    # sp=1: 14,8 -> 14,7
    # sp=2: 14,7 -> 14,6
    # sp=3: 14,6 -> 14,5
    # sp=4: 14,5 -> 14,4 -> 13,4
    # sp=5 (letzter sp wo discarded wird): 13,4 -> 13,3 -> 11,3
    # sp=6 (wird nur noch in final architecture umgewandelt): 11,3 -> 11,1 
    
    # how many channels are added
    if len(args.add_width) == 3:
        add_width = args.add_width
    else:
        add_width = [0, 0, 0, 0, 0, 0, 0]
        
    # how many layers are added
    if len(args.add_layers) == 3:
        add_layers = args.add_layers
    else:
        add_layers = [0, 0, 0, 0, 0, 0, 0]
        
    if len(args.dropout_rate) ==3:
        drop_rate = args.dropout_rate
    else:
        drop_rate = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
    # num of epochs without alpha weight updates    
    eps_no_archs = [1, 1, 1, 1, 1, 1, 1]
    
    epochs = [6, 6, 6, 6, 6, 6, 6]
    
    cnn_aux = 2
    rnn_aux = 6
    
    train_losses = []
    valid_losses = []
    acc_train = []
    acc_valid = []
    
    # iterate over stages
    for sp in range(len(num_to_keep)): 
        # sp=6
        
        if sp ==0:
         
           
           k_cnn = sum(1 for i in range(args.steps) for n in range(2+i))
          
           num_ops_cnn = len(switches_normal_cnn[0])
           
           alphas_normal = nn.Parameter(torch.FloatTensor(1e-3*np.random.randn(k_cnn, num_ops_cnn))) # vorher: k_cnn, num_ops_cnn
           alphas_reduce = nn.Parameter(torch.FloatTensor(1e-3*np.random.randn(k_cnn, num_ops_cnn))) # vorher: k_cnn, num_ops_cnn
           
           k_rhn = sum(i for i in range(1, rnn_steps+1))             
           num_ops_rhn = len(switches_rnn[0])
           
           alphas_rnn = nn.Parameter(torch.FloatTensor(1e-3*np.random.randn(k_rhn, num_ops_rhn)))
   
           model = one_shot_model.RNNModelSearch(args.seq_len, args.dropouth, args.dropoutx,
                              args.init_channels + int(add_width[sp]), args.num_classes, args.layers, args.steps, args.multiplier, args.stem_multiplier,  
                              True, 0.2, None, nn.CrossEntropyLoss().to(device), 
                              switches_normal_cnn, switches_reduce_cnn, switches_rnn, float(drop_rate[sp]), alphas_normal, alphas_reduce, alphas_rnn) 
      
        if sp > 0:
            
            old_dict = model.state_dict()
          
            new_reduce_arch = nn.Parameter(torch.FloatTensor(new_reduce_arch))
            new_normal_arch = nn.Parameter(torch.FloatTensor(new_normal_arch))
            new_rnn_arch = nn.Parameter(torch.FloatTensor(new_arch_rnn))
            
            model = one_shot_model.RNNModelSearch(args.seq_len, args.dropouth, args.dropoutx,
                           args.init_channels + int(add_width[sp]), args.num_classes, args.layers, args.steps, args.multiplier, args.stem_multiplier,  
                           True, 0.2, None, nn.CrossEntropyLoss().to(device), 
                           switches_normal_cnn, switches_reduce_cnn, switches_rnn, float(drop_rate[sp]), new_normal_arch, new_reduce_arch, new_rnn_arch) 

            new_dict = model.state_dict()
            # trained_weights = {k: v for k, v in old_dict.items() if k in new_dict}
           
            trained_weights = {k: v for k, v in old_dict.items() if k in new_dict}
            # müssen die old weights sein
           
            #trained_weights['stem.0.weight'] # 0.0064, (ist selbe wie in new_dict also falsch!!!)
            # jetzt aber mit 0.18 richtig
            trained_weights["alphas_normal"] = new_normal_arch 
            trained_weights["alphas_reduce"] = new_reduce_arch
            trained_weights["weights"] = new_rnn_arch
            trained_weights["rnns.0.weights"] = new_rnn_arch
            
            new_dict.update(trained_weights)

            model.load_state_dict(new_dict) 
            
            
    
        if args.cuda:
            if args.single_gpu:
                model = model.to(device)

            else:
                model = nn.DataParallel(model).to(device)

        else:
            model = model.to(device)
        
    
        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
        
        conv = []
        rhn = []
        for name, param in model.named_parameters():
            #print(name)
            #if 'stem' or 'preprocess' or 'conv' or 'bn' or 'fc' in name:
           if 'rnns' in name:
               #print(name)
               rhn.append(param)
           #elif 'decoder' in name:
           else:
               #print(name)
               conv.append(param)
        
        optimizer = torch.optim.SGD([{'params':conv}, {'params':rhn}], lr=args.cnn_lr, weight_decay=args.cnn_weight_decay)
        optimizer.param_groups[0]['lr'] = args.cnn_lr
        optimizer.param_groups[0]['weight_decay'] = args.cnn_weight_decay
        optimizer.param_groups[0]['momentum'] = args.momentum
        optimizer.param_groups[1]['lr'] = args.rhn_lr
        optimizer.param_groups[1]['weight_decay'] = args.rhn_weight_decay
        
    
        # optimizer for alpha updates
        optimizer_a = torch.optim.Adam(model.arch_parameters(), # model.module.arch_parameters()
                    lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(args.epochs), eta_min=args.learning_rate_min)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
        
        sm_dim = -1
        epochs = args.epochs
        eps_no_arch = eps_no_archs[sp]
        scale_factor = 0.2
        
        clip_params = [args.conv_clip, args.rhn_clip, args.clip]

        
        for epoch in range(epochs):
            
            train_start = time.strftime("%Y%m%d-%H%M")

            
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            logging.info('Epoch: %d lr: %e', epoch, lr)
            epoch_start = time.time()
            # training
            if epoch < eps_no_arch: 
                model.p = float(drop_rate[sp]) * (epochs - epoch - 1) / epochs 
                model.update_p()           
                train_acc, train_obj = train(train_object, valid_object, model, rhn, conv, criterion, optimizer, optimizer_a, lr, epoch, args.num_steps, clip_params, args.one_clip, args.report_freq, args.beta, train_arch=False)

            else:
                model.p = float(drop_rate[sp]) * np.exp(-(epoch - eps_no_arch) * scale_factor) 
                model.update_p()  
                train_acc, train_obj = train(train_object, valid_object, model, rhn, conv, criterion, optimizer, optimizer_a, lr, epoch, args.num_steps, clip_params, args.one_clip, args.report_freq, args.beta, train_arch=True)

            if args.validation == True:
                if epoch % args.report_validation == 0:
                    valid_acc, valid_obj = infer(valid_object, model, criterion, args.batch_size, args.num_steps, args.report_freq)
                    logging.info('Valid_acc %f', valid_acc)
                    
                    valid_losses.append(valid_obj)
                    acc_valid.append(valid_acc)
                    
            train_losses.append(train_obj)
            acc_train.append(train_acc)
            logging.info('Train_acc %f', train_acc)
            epoch_duration = time.time() - epoch_start
            logging.info('Epoch time: %ds', epoch_duration)
            # validation
            #if epochs - epoch < 5:
            #    valid_acc, valid_obj = infer(valid_object, model, criterion, args.batch_size)
            #    logging.info('Valid_acc %f', valid_acc)
        
        utils.save(model, os.path.join(args.save, 'weights.pt'))
        print('------Dropping %d paths------' % num_to_drop[sp])
        # Save switches info for s-c refinement. 
        if sp == len(num_to_keep) - 1: 
            switches_normal_2 = copy.deepcopy(switches_normal_cnn)
            switches_reduce_2 = copy.deepcopy(switches_reduce_cnn)
            switches_rnn2 = copy.deepcopy(switches_rnn)
            
            
        new_normal_arch, new_reduce_arch, switches_normal_cnn, switches_reduce_cnn = discard_cnn_ops(model, switches_normal_cnn, switches_reduce_cnn, num_to_keep, num_to_drop, sp, new_alpha_values=True)
        
        
            
        if sp in disc_rhn_ops[0]:
            
            new_arch_rnn, switches_rnn = discard_rhn_ops(model, switches_rnn, num_to_keep_rnn, num_to_drop_rnn, sp, new_alpha_values=True)

        
        # (haben 7 stages: sp=0:6)
        # in sp=5: discarde bei i=2 1ne edge
        # in sp=6: discarde bei i=1 1ne edge und bei i=2 1ne edge
        
        ## discard CNN edges ##
        #if 3 < sp < 6: 
        if sp in disc_cnn_idxs[0]:
            
            # currently I only implemented for switches_normal_cnn, so I have to do this, also for switches_reduce_cnn
            switches_normal_cnn = copy.deepcopy(switches_normal_cnn)
            switches_reduce_cnn = copy.deepcopy(switches_reduce_cnn)
            
            new_red_arch = F.softmax(new_reduce_arch, dim=-1).data.cpu().numpy()
            new_nor_arch = F.softmax(new_normal_arch, dim=-1).data.cpu().numpy()
          
            final_prob_normal = [0.0 for i in range(len(switches_normal_cnn))]
            final_prob_reduce = [0.0 for i in range(len(switches_reduce_cnn))]
        
            switch_count_normal = 0
            switch_count_reduce = 0

            for i in range(len(switches_normal_cnn)): 
                if switches_normal_cnn[i].count(False) != len(switches_normal_cnn[i]): #wenn nicht alle false sind -> keine discarded edge
                    if switches_normal_cnn[i][0] ==True:
                        new_nor_arch[i-switch_count_normal][0] = 0
                    final_prob_normal[i] = max(new_nor_arch[i-switch_count_normal]) # add best/max operation to final_prob 
                else: # if discarded edge
                    final_prob_normal[i] = 1 # set previous discarded edge to 1 (so that it does not get discarded in this stage)
                    switch_count_normal += 1
                    
                if switches_reduce_cnn[i].count(False) != len(switches_reduce_cnn[i]): # if no discarded edge
                    if switches_reduce_cnn[i][0] ==True:
                        new_red_arch[i-switch_count_reduce][0]=0
                    final_prob_reduce[i] = max(new_red_arch[i-switch_count_reduce])
                else: # if discarded edge
                    final_prob_reduce[i] = 1
                    switch_count_reduce += 1
                    
                      
            discard_switch_normal = [] 
            discard_switch_reduce = []
            n = 3
            start = 2
            
            for i in range(3):
                # i=0
                # print(i): 0,1,2
                # which_node: bei sp4 nur 3ter, bei sp5 2ter und 3ter, bei sp6 1ter, 2ter und 3ter
                # 4-3=1; 
                # 4-1=3, 4-2=2, 4-3=1
                # 5-1=4, 5-2=3, 5-3=2
                # i =2
                end = start + n

    
                if i >= cnn_aux: # d.h. bei i=0 und i=1 passiert nichts, sondern nur bei i=2 (d.h. er discarded von letztem Node also zwischen 9 und 13 row von switches)
                # sorgt dafür, dass bei 1tem discard_edge durchlauf (sp=5) 
                    edges_normal = final_prob_normal[start:end]
                    edges_reduce = final_prob_reduce[start:end]
                    
                    edge_n = sorted(range(n), key=lambda x: edges_normal[x]) # sortiert nach, von schlechtem zu guten
                    discard_edge_n = edge_n[0] + start # schlechteste von allen ist 0tes element, deshalb edge[0]
                    discard_switch_normal.append(discard_edge_n)
                    
                    edge_r = sorted(range(n), key=lambda x: edges_reduce[x]) # sortiert nach, von schlechtem zu guten
                    discard_edge_r = edge_r[0] + start # schlechteste von allen ist 0tes element, deshalb edge[0]
                    discard_switch_reduce.append(discard_edge_r)
                    
                    cnn_aux -= 1 # damit bei nächster discard_edge durchlauf (sp=6) schon bei i=1 und bei i=2 was passiert
                
                start = end
                n = n + 1
            
        
            #new_normal_arch_aux = new_normal_arch
            #new_reduce_arch_aux = new_reduce_arch
            
            num_edges = len(new_normal_arch)
            
            num_discard_n, num_discard_r = 0, 0
            
            switch_count_normal = 0
            switch_count_reduce = 0
                
            for i in range(1,len(switches_normal_cnn)):
                # i=0
                if i in discard_switch_normal: # if an edge should be discarded in this stage
                    
                    num_discard_n += 1
                                   
                    new_normal_arch = new_normal_arch[new_normal_arch!=new_normal_arch[i-switch_count_normal]].view(num_edges-num_discard_n, num_to_keep[sp])
                    
                    for j in range(len(PRIMITIVES_cnn)):
                        switches_normal_cnn[i][j] = False 
                        
                        
                if switches_normal_cnn[i].count(False) == len(switches_normal_cnn[i]): # if there is a discarded edge from previous stage
            
                    switch_count_normal += 1


                if i in discard_switch_reduce: 
                    
                    num_discard_r += 1
                    
                    new_reduce_arch = new_reduce_arch[new_reduce_arch!=new_reduce_arch[i-switch_count_reduce]].view(num_edges-num_discard_r, num_to_keep[sp])

                    for j in range(len(PRIMITIVES_cnn)):
                        switches_reduce_cnn[i][j] = False  
                        
                        
                if switches_reduce_cnn[i].count(False) == len(switches_reduce_cnn[i]):
            
                    switch_count_reduce += 1
                    
                    
        if sp in disc_rhn_idxs[0]:
            # currently I only implemented for switches_normal_cnn, so I have to do this, also for switches_reduce_cnn
            switches_rnn = copy.deepcopy(switches_rnn)
    
            rnn_final = [0 for idx in range(36)]
            # probs_out = copy.deepcopy(probs_in)

            new_arch = F.softmax(new_arch_rnn, dim=-1).data.cpu().numpy()
        
            switch_count_rnn = 0

            for i in range(len(switches_rnn)): 
                # i=0
                if switches_rnn[i].count(False) != len(switches_rnn[i]): #wenn nicht alle false sind -> keine discarded edge
                    if switches_rnn[i][0] ==True:
                        new_arch[i-switch_count_rnn][0] = 0
                    # d.h. wir bekommen die 14 max values raus von jeder edge: jetzt würde ich einfach 
                    # erstmal von dem letzten Node (welche 5 edges hat) die schlechteste discarden, indem wir die gesamte switches 
                    # Zeile False setzten
                    rnn_final[i] = max(new_arch[i-switch_count_rnn]) # add best/max operation to final_prob 
                else: # if previous discarded edge
                    rnn_final[i] = 1 # set previous discarded edge to 1 (so that it does not get discarded in this stage)
                    switch_count_rnn += 1
            
                
            start = 1
            discard_switch_rnn=[]
            n=2
            for i in range(7):
                
                 end = start + i + 2 
                 
                 if i >= rnn_aux:
                     
                     tbrnn = rnn_final[start:end]

                     # bei sp=1 nur bei letztem iter also i=6 zwischen [28,35] discarden
                     # bei sp=2 bei beiden letzten iters also i=5 und i=6 also zwischen [21,27] und [28,35] jeweils 1ne edge discarden
                 
                     edge_rnn = sorted(range(n), key=lambda x: tbrnn[x]) 

                     discard_switch_rnn.append(edge_rnn[0] + start)
            
                 start = end
                 n +=1
                 
            rnn_aux -=1
            
            
            num_edges = len(new_arch_rnn)
            num_discard = 0
            switch_count_rnn = 0
            for i in range(1,len(switches_rnn)):
                # i=30
                if i in discard_switch_rnn: # if an edge should be discarded in this stage
                    
                    num_discard += 1
                                   
                    new_arch_rnn = new_arch_rnn[new_arch_rnn!=new_arch_rnn[i-switch_count_rnn]].view(num_edges-num_discard, num_to_keep_rnn[sp])
                    
                    for j in range(len(PRIMITIVES_rnn)):
                        switches_rnn[i][j] = False 
                        
                        
                if switches_rnn[i].count(False) == len(switches_rnn[i]): # if there is a discarded edge from previous stage
            
                    switch_count_rnn += 1
                    
                
        # get genotype    
        if sp != len(num_to_keep) - 1:
            genotype = parse_genotype(switches_normal_cnn, new_normal_arch, switches_reduce_cnn, new_reduce_arch, switches_rnn, new_arch_rnn)



        if sp == len(num_to_keep) - 1: 
           
            genotype, switches_normal_cnn, switches_reduce_cnn, switches_rnn, normal_prob, reduce_prob, rnn_prob = final_stage_genotype(model, switches_normal_cnn, switches_normal_2, switches_reduce_cnn, switches_reduce_2, switches_rnn, switches_rnn2)

          
            logging.info(genotype)
            logging.info('Restricting skipconnect...')
            # regularization of skip connections
            for sks in range(0, 9): 
                # sks=0
                max_sk = 8 - sks                
                num_sk = check_sk_number(switches_normal_cnn) # counts number of identity/scip connections, 2
               
                if not num_sk > max_sk: # 2 > 8 for i=0 continue, 2 > 1
                    continue
                while num_sk > max_sk: # starts with 2>1
                    normal_prob = delete_min_sk_prob(switches_normal_cnn, switches_normal_2, normal_prob)
                    switches_normal_cnn = keep_1_on(switches_normal_2, normal_prob) # von aktuell 4 übrig gebliebenen, werden jetzt nochmal 2 gekickt, haben also 2 übrig
                    switches_normal_cnn = keep_2_branches(switches_normal_cnn, normal_prob)
                    num_sk = check_sk_number(switches_normal_cnn)
                logging.info('Number of skip-connect: %d', max_sk)
                genotype = parse_network(switches_normal_cnn, switches_reduce_cnn, switches_rnn)
                logging.info(genotype) 
                
            genotype_file = '{}-pdarts_geno'.format(args.save)
            np.save(genotype_file, genotype)
            
            trainloss_file = '{}-train_loss-{}'.format(args.save, train_start)
            np.save(trainloss_file, train_losses)
      
            acctrain_file = '{}-acc_train-{}'.format(args.save, train_start) 
            np.save(acctrain_file, acc_train)
            
            
            validloss_file = '{}-valid_loss-{}'.format(args.save, train_start)
            np.save(validloss_file, valid_losses)
      
            accvalid_file = '{}-acc_valid-{}'.format(args.save, train_start) 
            np.save(acctrain_file, acc_valid)
         


if __name__ == '__main__':
    start_time = time.time()
    main() 
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Total searching time: %ds', duration)
