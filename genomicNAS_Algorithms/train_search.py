#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 10:24:24 2021

@author: amadeu
"""

import argparse
import os, sys, glob
import time
import math
import numpy as np
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from darts_tools.architect import Architect

import torch.utils

import copy
import gc
#import model_search as one_shot_model
import model_search as one_shot_model

from generalNAS_tools.genotypes import PRIMITIVES_cnn, PRIMITIVES_rnn, rnn_steps, CONCAT, Genotype

import generalNAS_tools.data_preprocessing_new as dp

# sys.path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from generalNAS_tools.utils import repackage_hidden, create_exp_dir, save_checkpoint 
import generalNAS_tools.utils
from generalNAS_tools.train_and_validate import train, infer

from generalNAS_tools.utils import repackage_hidden, create_exp_dir, save_checkpoint 
from generalNAS_tools import utils

from darts_tools.final_stage_run import final_stage_genotype
from darts_tools.auxiliary_functions import *
from darts_tools.discard_operations import discard_cnn_ops, discard_rhn_ops

parser = argparse.ArgumentParser(description='DARTS for genomic Data')
parser.add_argument('--data', type=str, default='/home/amadeu/anaconda3/envs/darts_env/cnn/data2/trainset.txt', help='location of the data corpus')
parser.add_argument('--cnn_lr', type=float, default=0.025, help='learning rate for CNN part')
parser.add_argument('--cnn_weight_decay', type=float, default=3e-4, help='weight decay for CNN part')
parser.add_argument('--rhn_lr', type=float, default=2, help='learning rate for RHN part')
parser.add_argument('--rhn_weight_decay', type=float, default=5e-7, help='weight decay for RHN part')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--num_steps', type=int, default=2, help='number of iterations per epoch')
parser.add_argument('--valid_directory', type=str, default='/home/amadeu/Downloads/genomicData/validation', help='directory of validation data')
parser.add_argument('--train_directory', type=str, default='/home/amadeu/Downloads/genomicData/train', help='directory of training data')
parser.add_argument('--num_files', type=int, default=3, help='number of files for training data')

parser.add_argument('--validation', type=bool, default=True)
parser.add_argument('--report_validation', type=int, default=2, help='validation epochs') 

parser.add_argument('--next_character_prediction', type=bool, default=True, help='task of model')
parser.add_argument('--one_clip', type=bool, default=True)
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--conv_clip', type=float, default=5, help='gradient clipping of convs')
parser.add_argument('--rhn_clip', type=float, default=0.25, help='gradient clipping of lstms')

parser.add_argument('--seq_len', type=int, default=1000, help='sequence length')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--init_channels', type=int, default=8, help='num of init channels')
parser.add_argument('--layers', type=int, default=6, help='total number of layers')
parser.add_argument('--num_classes', type=int, default=4, help='num of output classes') 
parser.add_argument('--steps', type=int, default=4, help='total number of Nodes')
parser.add_argument('--multiplier', type=int, default=4, help='multiplier')
parser.add_argument('--stem_multiplier', type=int, default=3, help='stem multiplier')

parser.add_argument('--epochs', type=int, default=2,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                    help='batch size')
parser.add_argument('--seq_size', type=int, default=20,
                    help='sequence length')

parser.add_argument('--dropouth', type=float, default=0.25,
                    help='dropout for hidden nodes in rnn layers (0 = no dropout)')
parser.add_argument('--dropoutx', type=float, default=0.75,
                    help='dropout for input nodes in rnn layers (0 = no dropout)')

parser.add_argument('--seed', type=int, default=3,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--report_freq', type=int, default=50, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='EXP',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=0,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1e-3,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=5e-7,
                    help='weight decay applied to all weights')
parser.add_argument('--continue_train', action='store_true',
                    help='continue train from a checkpoint')
parser.add_argument('--small_batch_size', type=int, default=-1,
                    help='the batch size for computation. batch_size should be divisible by small_batch_size.\
                     In our implementation, we compute gradients with small_batch_size multiple times, and accumulate the gradients\
                     until batch_size is reached. An update step is then performed.')
parser.add_argument('--max_seq_len_delta', type=int, default=20,
                    help='max sequence length')
parser.add_argument('--single_gpu', default=True, action='store_false', 
                    help='use single GPU')
parser.add_argument('--gpu', type=int, default=0, help='GPU device to use')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_wdecay', type=float, default=1e-3,
                    help='weight decay for the architecture encoding alpha')
parser.add_argument('--arch_lr', type=float, default=3e-3,
                    help='learning rate for the architecture encoding alpha')
parser.add_argument('--note', type=str, default='try', help='note for this run')

#parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
#parser.add_argument('--search', type=bool, default=True, help='which architecture to use')
#parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
#parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
args = parser.parse_args()


args.save = '{}search-{}-{}'.format(args.save, args.note, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)



np.random.seed(args.seed)
torch.manual_seed(args.seed)
  
logging.info("args = %s", args)
       
train_object, valid_object, num_classes = dp.data_preprocessing(train_directory = args.train_directory, valid_directory = args.valid_directory, num_files=args.num_files,
        seq_size = args.seq_len, batch_size=args.batch_size, next_character=True)

_, valid_data, num_classes = dp.data_preprocessing(train_directory = args.train_directory, valid_directory = args.valid_directory, num_files=args.num_files,
        seq_size = args.seq_len, batch_size=args.batch_size, next_character=True)
    
# build Network
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)


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


# initialize alpha weights
k_cnn = sum(1 for i in range(args.steps) for n in range(2+i))
      
num_ops_cnn = sum(switches_normal_cnn[0])
       
alphas_normal = nn.Parameter(torch.FloatTensor(1e-3*np.random.randn(k_cnn, num_ops_cnn))) # vorher: k_cnn, num_ops_cnn
alphas_reduce = nn.Parameter(torch.FloatTensor(1e-3*np.random.randn(k_cnn, num_ops_cnn))) # vorher: k_cnn, num_ops_cnn
       
k_rhn = sum(i for i in range(1, rnn_steps+1))             
num_ops_rhn = sum(switches_rnn[0])
       
alphas_rnn = nn.Parameter(torch.FloatTensor(1e-3*np.random.randn(k_rhn, num_ops_rhn)))

multiplier, stem_multiplier = 4, 3

#num_motifs=300
# import math

# alphas_rnn[(1, 3, 8, 14, 18, 20),3]= -1000



if args.continue_train:
    model = torch.load(os.path.join(args.save, 'model.pt'))
else:
    #model = one_shot_model.RNNModelSearch(args.seq_len,# args.emsize, args.nhid, args.nhidlast,
    #                       args.dropouth, args.dropoutx,
    #                       args.init_channels, args.num_classes, args.layers, args.steps, num_motifs) 
    
    model = one_shot_model.RNNModelSearch(args.seq_len, args.dropouth, args.dropoutx,
                              args.init_channels, args.num_classes, args.layers, args.steps, multiplier, stem_multiplier,  
                              True, 0.2, None, nn.CrossEntropyLoss().to(device), 
                              switches_normal_cnn, switches_reduce_cnn, switches_rnn, 0.0, alphas_normal, alphas_reduce, alphas_rnn) 
   
    
size = 0
for p in model.parameters():
    size += p.nelement()
logging.info('param size: {}'.format(size))
logging.info('initial genotype:')

#if args.cuda:
#    if args.single_gpu:
#        parallel_model = model.to(device)
#    else:
#        parallel_model = nn.DataParallel(model, dim=1).to(device)
#else:
    
parallel_model = model.to(device)
        
#optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)


#for s,k in model.parameters():
#    print()
    
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
    
   
    
logging.info(model.genotype())
architect = Architect(parallel_model, args)
    

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    
sm_dim = -1
epochs = args.epochs
    #eps_no_arch = eps_no_archs[sp]
scale_factor = 0.2
    
architect = Architect(model, args)

train_losses = []
valid_losses = []
acc_train = []
acc_valid = []
    
clip_params = [args.conv_clip, args.rhn_clip, args.clip]



for epoch in range(epochs):
    # epoch=0
    train_start = time.strftime("%Y%m%d-%H%M")
   
    lr = scheduler.get_last_lr()[0]
    logging.info('Epoch: %d lr: %e', epoch, lr)
    epoch_start = time.time()

    train_acc, train_obj = train(train_object, valid_object, model, rhn, conv, criterion, optimizer, None, architect, args.unrolled, lr, epoch, args.num_steps, clip_params, args.report_freq, args.beta, args.one_clip, train_arch=True, pdarts=False)
    
    scheduler.step()

    
    # validation
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
    
    genotype = model.genotype()
    logging.info(genotype) 
            
    
    
genotype_file = '{}-darts_geno'.format(args.save)
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

