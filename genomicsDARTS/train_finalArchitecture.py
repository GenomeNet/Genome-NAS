#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 13:29:57 2021

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
from architect import Architect

import genotypes_rnn


import gc

import data
import model_search as model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from utils import batchify, get_batch, repackage_hidden, create_exp_dir, save_checkpoint


import data_preprocessing as dp


parser = argparse.ArgumentParser(description='Evaluate final architecture found by DARTS')
parser.add_argument('--data', type=str, default='/home/amadeu/anaconda3/envs/darts_env/cnn/data2/trainset.txt', help='location of the data corpus')

parser.add_argument('--emsize', type=int, default=128,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=128,
                    help='number of hidden units per layer')
parser.add_argument('--nhidlast', type=int, default=128,
                    help='number of hidden units for the last rnn layer')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--init_channels', type=int, default=8, help='num of init channels') # args.C, args.num_classes, args.layers, args.steps=4, args.multiplier=4, args.stem_multiplier=3
parser.add_argument('--layers', type=int, default=3, help='total number of layers')
parser.add_argument('--num_classes', type=int, default=4, help='num of output classes') # args.C, args.num_classes, args.layers, args.steps=4, args.multiplier=4, args.stem_multiplier=3
parser.add_argument('--steps', type=int, default=4, help='total number of Nodes')
parser.add_argument('--multiplier', type=int, default=4, help='multiplier')
parser.add_argument('--stem_multiplier', type=int, default=3, help='stem multiplier')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=2,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=10,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.75,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.25,
                    help='dropout for hidden nodes in rnn layers (0 = no dropout)')
parser.add_argument('--dropoutx', type=float, default=0.75,
                    help='dropout for input nodes in rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.2,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--seed', type=int, default=3,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
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
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--search', type=bool, default=False, help='which architecture to use')
#parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
#parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')

args = parser.parse_args()


# import data
# '/home/amadeu/anaconda3/envs/darts_env/cnn/data2/trainset.txt'
# '/home/ascheppa/miniconda2/envs/darts/cnn/data2/trainset.txt'

train_object, valid_object, num_classes = dp.data_preprocessing(data_file =args.data, 
          seq_size = args.bptt, representation = 'onehot', model_type = 'CNN', batch_size=args.batch_size)


eval_batch_size = 10

_, valid_data, num_classes = dp.data_preprocessing(data_file =args.data, 
          seq_size = args.bptt, representation = 'onehot', model_type = 'CNN', batch_size=eval_batch_size)


if args.nhidlast < 0:
    args.nhidlast = args.emsize
if args.small_batch_size < 0:
    args.small_batch_size = args.batch_size

if not args.continue_train:
    args.save = 'search-{}'.format(args.save)
    create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

np.random.seed(args.seed)
torch.manual_seed(args.seed)



if args.continue_train:
    model = torch.load(os.path.join(args.save, 'model.pt'))
else:      
    genotype = eval("genotypes_rnn.%s" % args.arch)

    model = model.RNNModel(args.bptt, args.emsize, args.nhid, args.nhidlast, 
                        args.dropout, args.dropouth, args.dropoutx, args.dropouti, args.dropoute, 
                        args.init_channels, args.num_classes, args.layers, args.steps, args.multiplier, args.stem_multiplier,
                        args.search, args.drop_path_prob, genotype=genotype)
#args.search=False
#weights = []
#for v in model.arch_parameters():
#    weights.append(v)
#    print(v.shape)
#weights = weights[2]
#for k,v in model.named_parameters():
#    print(k)
    


size = 0
for p in model.parameters():
    size += p.nelement()
logging.info('param size: {}'.format(size))
logging.info('initial genotype:')
#logging.info(model.genotype())


if args.cuda:
    if args.single_gpu:
        #parallel_model = model.cuda()
        parallel_model = model.to(device)

    else:
        #parallel_model = nn.DataParallel(model, dim=1).cuda()
        parallel_model = nn.DataParallel(model, dim=1).to(device)

else:
    parallel_model = model
    

#architect = Architect(parallel_model, args)

total_params = sum(x.data.nelement() for x in model.parameters())
logging.info('Args: {}'.format(args))
logging.info('Model total parameters: {}'.format(total_params))

# model.genotype() 

def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    #ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    #for i in range(0, data_source.size(0) - 1, args.bptt):
    for step, (input, target) in enumerate(valid_data):

        #data, targets = get_batch(data_source, i, args, evaluation=True)
        data, targets = input, target
        
        #targets = targets.view(-1)
        targets = torch.max(targets, 1)[1]


        log_prob, hidden = parallel_model(data, hidden)
        criterion = nn.CrossEntropyLoss()

        loss = criterion(log_prob, targets)
        
        total_loss += loss * len(data)

        hidden = repackage_hidden(hidden)
        
    return total_loss.item() / len(data_source)


def train(epoch):
    assert args.batch_size % args.small_batch_size == 0, 'batch_size must be divisible by small_batch_size'

    # Turn on training mode which enables dropout.
    total_loss = 0
    start_time = time.time()
    hidden = [model.init_hidden(args.small_batch_size) for _ in range(args.batch_size // args.small_batch_size)]
   
    # sollte so passen mit dieser for-loop, weil originales DARTS macht 1te while loop selbe wie unsere for-loop und seine 2te while loop
    # macht nur Sinn falls small_batch_size=!batch_size ansonsten wird dort nach jeder iteration start=end(=batch_size) gesetzt, was bedeutet
    # dass er wieder zur 1ten while loop springt und dort wird s_id wieder auf 0 gesetzt!!! s_id bleibt also immer 0 !!!!
    for step, (input, target) in enumerate(train_object):
       
        data, targets = input, target
        data_valid, targets_valid = next(iter(valid_object))
        
        targets = torch.max(targets, 1)[1]
        targets_valid = torch.max(targets_valid, 1)[1]
        
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
     
        seq_len = int(bptt)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()

        optimizer.zero_grad()

        s_id = 0
    
        cur_data, cur_targets = data, targets

      
        hidden[s_id] = repackage_hidden(hidden[s_id])

        #optimizer.zero_grad()
        
        # hidden[s_id][0].shape ist [1,2,50] weil 1 timestep, bei 2 batches und 50 output hidden units von dem RNN
        # cur_data.shape ist [10,2], weil 10 timesteps, bei 2 batches
        log_prob, hidden[s_id], rnn_hs, dropped_rnn_hs = parallel_model(cur_data, hidden[s_id], return_h=True)
        #raw_loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), cur_targets)
        criterion = nn.CrossEntropyLoss()

        raw_loss = criterion(log_prob, cur_targets)

        loss = raw_loss
        # Activiation Regularization
        #if args.alpha > 0: # per default args.alpha=0, so we don't need this part
        #  loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss *= args.small_batch_size / args.batch_size
        total_loss += raw_loss.data * args.small_batch_size / args.batch_size
        loss.backward()
        
        #for rnn_h in rnn_hs[-1:]:
        #    print(rnn_h.shape)


        gc.collect()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        # total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            logging.info(parallel_model.genotype())
            logging.info(F.softmax(parallel_model.weights, dim=-1))
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            logging.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        batch += 1
        i += seq_len

    logging.info('{} epoch done   '.format(epoch) + str(parallel_model.genotype()))

# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000

if args.continue_train:
    optimizer_state = torch.load(os.path.join(args.save, 'optimizer.pt'))
    if 't0' in optimizer_state['param_groups'][0]:
        optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    optimizer.load_state_dict(optimizer_state)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)

for epoch in range(1, args.epochs+1):
    # epoch=1
    epoch_start_time = time.time()
    train(epoch)

    val_loss = evaluate(val_data, eval_batch_size)
    logging.info('-' * 89)
    logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                       val_loss, math.exp(val_loss)))
    logging.info('-' * 89)

    if val_loss < stored_loss:
        save_checkpoint(model, optimizer, epoch, args.save)
        logging.info('Saving Normal!')
        stored_loss = val_loss

    best_val_loss.append(val_loss)