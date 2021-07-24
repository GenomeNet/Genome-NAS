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
import darts_tools.model_search as one_shot_model
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

from generalNAS_tools.utils import scores_perClass, scores_Overall, pr_aucPerClass, roc_aucPerClass, overall_acc, overall_f1

from generalNAS_tools.utils import scores_perClass, scores_Overall, pr_aucPerClass, roc_aucPerClass, overall_acc, overall_f1





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
parser.add_argument('--test_directory', type=str, default='/home/amadeu/Downloads/genomicData/test', help='directory of test data')

parser.add_argument('--num_files', type=int, default=3, help='number of files for training data')

parser.add_argument('--train_input_directory', type=str, default='/home/amadeu/Desktop/GenomNet_MA/data/inputs_small.pkl', help='directory of train data')
parser.add_argument('--train_target_directory', type=str, default='/home/amadeu/Desktop/GenomNet_MA/data/targets_small.pkl', help='directory of train data')
parser.add_argument('--valid_input_directory', type=str, default='/home/amadeu/Desktop/GenomNet_MA/data/inputs_small_val.pkl', help='directory of validation data')
parser.add_argument('--valid_target_directory', type=str, default='/home/amadeu/Desktop/GenomNet_MA/data/targets_small_val.pkl', help='directory of validation data')
parser.add_argument('--test_input_directory', type=str, default='/home/amadeu/Desktop/GenomNet_MA/data/inputs_small_test.pkl', help='directory of test data')
parser.add_argument('--test_target_directory', type=str, default='/home/amadeu/Desktop/GenomNet_MA/data/targets_small_test.pkl', help='directory of test data')

parser.add_argument('--task', type=str, default='next_character_prediction', help='defines the task')#TF_bindings

parser.add_argument('--validation', type=bool, default=True)
parser.add_argument('--report_validation', type=int, default=2, help='validation epochs') 

parser.add_argument('--next_character_prediction', type=bool, default=True, help='task of model')
parser.add_argument('--one_clip', type=bool, default=True)
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--conv_clip', type=float, default=5, help='gradient clipping of convs')
parser.add_argument('--rhn_clip', type=float, default=0.25, help='gradient clipping of lstms')

parser.add_argument('--seq_size', type=int, default=200, help='sequence length') # 1000
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--init_channels', type=int, default=8, help='num of init channels')
parser.add_argument('--layers', type=int, default=3, help='total number of layers')
# parser.add_argument('--num_classes', type=int, default=919, help='num of output classes') 
parser.add_argument('--steps', type=int, default=4, help='total number of Nodes')
#parser.add_argument('--multiplier', type=int, default=4, help='multiplier')
#parser.add_argument('--stem_multiplier', type=int, default=3, help='stem multiplier')
parser.add_argument('--epochs', type=int, default=2,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                    help='batch size')

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
parser.add_argument('--report_freq', type=float, default=2, help='report frequency')

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
args = parser.parse_args()


args.save = '{}search-{}-{}'.format(args.save, args.note, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
#logging.getLogger().addHandler(fh)

logging = logging.getLogger(__name__)


def main():

    # np.random.seed(args.seed)
    torch.manual_seed(args.seed)
      
    logging.info("args = %s", args)
           
    #train_object, valid_object, num_classes = dp.data_preprocessing(train_directory = args.train_directory, valid_directory = args.valid_directory, num_files=args.num_files,
    #        seq_size = args.seq_len, batch_size=args.batch_size, next_character=args.next_character_prediction)
    
    #_, valid_data, num_classes = dp.data_preprocessing(train_directory = args.train_directory, valid_directory = args.valid_directory, num_files=args.num_files,
    #        seq_size = args.seq_len, batch_size=args.batch_size, next_character=args.next_character_prediction)
    
    if args.task == "next_character_prediction" or "sequence_to_sequence":
        
        import generalNAS_tools.data_preprocessing_new as dp

        train_queue, valid_queue, num_classes = dp.data_preprocessing(train_directory = args.train_directory, valid_directory = args.valid_directory, num_files=args.num_files,
                seq_size = args.seq_size, batch_size=args.batch_size, next_character=args.next_character_prediction)
      
        _, test_queue, _ = dp.data_preprocessing(train_directory = args.train_directory, valid_directory = args.test_directory, num_files=args.num_files,
                seq_size = args.seq_size, batch_size=args.batch_size, next_character=args.next_character_prediction)
        
        criterion = nn.CrossEntropyLoss().to(device)
        
    if args.task == 'TF_bindings':
        
        import generalNAS_tools.data_preprocessing_TF as dp
        
        train_queue, valid_queue, test_queue = dp.data_preprocessing(args.train_input_directory, args.valid_input_directory, args.test_input_directory, args.train_target_directory, args.valid_target_directory, args.test_target_directory, args.batch_size)
      
        criterion = nn.BCELoss().to(device)

        num_classes = 919
    
        
    # build Network
    
    
    
    # initialize switches
    # in DARTS they should always be True for all operations, because we keep all operations and edges during
    # the search process
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
    
    multiplier, stem_multiplier = 4,3
    

    model = one_shot_model.RNNModelSearch(args.seq_size, args.dropouth, args.dropoutx,
                              args.init_channels, num_classes, args.layers, args.steps, multiplier, stem_multiplier,  
                              True, 0.2, None, args.task, 
                              switches_normal_cnn, switches_reduce_cnn, switches_rnn, 0.0, alphas_normal, alphas_reduce, alphas_rnn) 
                                  # float(drop_rate[sp])) 
                                  
    # old_dict = model.state_dict()

        
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
    #optimizer_a = torch.optim.Adam(model.arch_parameters(), # model.module.arch_parameters()
    #            lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
    
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
    
    train_acc = []
    #all_predictions_train = []
    
    valid_acc = []
    #all_predictions_valid = []
    
    time_per_epoch = []
    
    
    clip_params = [args.conv_clip, args.rhn_clip, args.clip]
    
        
    for epoch in range(epochs):
            # epoch=0
            train_start = time.strftime("%Y%m%d-%H%M")
   
            lr = scheduler.get_last_lr()[0]
            logging.info('Epoch: %d lr: %e', epoch, lr)
            epoch_start = time.time()

            labels, predictions, train_loss = train(train_queue, valid_queue, model, rhn, conv, criterion, optimizer, None, architect, args.unrolled, lr, epoch, args.num_steps, clip_params, args.report_freq, args.beta, args.one_clip, train_arch=True, pdarts=False, task=args.task)
            
            scheduler.step()
            
            labels = np.concatenate(labels)
            predictions = np.concatenate(predictions)
            
            if args.task == 'next_character_prediction':
                acc = overall_acc(labels, predictions, args.task)
                logging.info('| epoch {:3d} | train acc {:5.2f}'.format(epoch, acc))
                train_acc.append(acc)


            else:
                f1 = overall_f1(labels, predictions, args.task)
                logging.info('| epoch {:3d} | train f1-score {:5.2f}'.format(epoch, f1))
                train_acc.append(f1)

     
      
            #all_labels_train.append(labels)
            #all_predictions_train.append(predictions)
            train_losses.append(train_loss)
            epoch_end = time.time()
            time_per_epoch.append(epoch_end)

            
            # validation
            if args.validation == True:
                if epoch % args.report_validation == 0:
                    labels, predictions, valid_obj = infer(valid_queue, model, criterion, args.batch_size, args.num_steps, args.report_freq, task=args.task)
                    logging.info('Valid_acc %f', valid_acc)
                    
                    labels = np.concatenate(labels)
                    predictions = np.concatenate(predictions)
                    
                    valid_losses.append(valid_loss)
                    #all_labels_valid.append(labels)
                    #all_predictions_valid.append(predictions)
            
                    if args.task == 'next_character_prediction':
                        acc = overall_acc(labels, predictions, args.task)
                        logging.info('| epoch {:3d} | valid acc {:5.2f}'.format(epoch, acc))
                        valid_acc.append(acc)

                    else:
                        f1 = overall_f1(labels, predictions, args.task)
                        logging.info('| epoch {:3d} | valid f1-score {:5.2f}'.format(epoch, f1))
                        valid_acc.append(f1)
            
          
            
            epoch_end = time.time()
            epoch_duration = epoch_end - epoch_start
            logging.info('Epoch time: %ds', epoch_duration)
            time_per_epoch.append(epoch_end)
            
            genotype = model.genotype()
            logging.info(genotype) 
            
    
    
    genotype_file = '{}-darts_geno'.format(args.save)
    np.save(genotype_file, genotype)
  
    trainloss_file = '{}-train_loss-{}'.format(args.save, train_start)
    np.save(trainloss_file, train_losses)
    acc_train_file = '{}-labels_train-{}'.format(args.save, train_start)
    np.save(acc_train_file, train_acc)
    #predictions__train_file = '{}-predictions_train-{}'.format(args.save, train_start)
    #np.save(predictions__train_file, all_predictions_train)
    
    time_file = '{}-time-{}'.format(args.save, train_start)
    np.save(time_file, time_per_epoch)
      

    # safe valid data
    validloss_file = '{}-valid_loss-{}'.format(args.save, train_start)
    np.save(validloss_file, valid_losses)
    acc_valid_file = '{}-labels_valid-{}'.format(args.save, train_start)
    np.save(acc_valid_file, valid_acc)
    #predictions__valid_file = '{}-predictions_valid-{}'.format(args.save, train_start)
    #np.save(predictions__valid_file, all_predictions_valid)
    




#def train(epoch):
    
#    objs = utils.AvgrageMeter()
#    top1 = utils.AvgrageMeter()
#    top5 = utils.AvgrageMeter()
#    total_loss = 0
#    start_time = time.time()
 
    
#    for step, (input, target) in enumerate(train_object):

#        if step > args.num_steps:
#            break
        
#        s_id = 0
        # input, labels for model parameter updates
#        data, targets = input, target
        
        # input, labels for arch parameter updates
#        data_valid, targets_valid = next(iter(valid_object))
        
#        model.train()

        # get correct shape of labels
#        targets = torch.max(targets, 1)[1]
#        targets_valid = torch.max(targets_valid, 1)[1]  
        
        # for stateless RHN, reset hidden state after each epoch
        #hidden[s_id] = repackage_hidden(hidden[s_id])
        #hidden_valid[s_id] = repackage_hidden(hidden_valid[s_id])
        
#        hidden = model.init_hidden(args.batch_size)
#        hidden_valid = model.init_hidden(args.batch_size)


        
#        cur_data, cur_data_valid = data.transpose(1, 2).float().to(device), data_valid.transpose(1, 2).float().to(device)
#        cur_targets, cur_targets_valid = targets.to(device), targets_valid.to(device)



#        hidden_valid, grad_norm = architect.step( 
#                hidden, cur_data, cur_targets,
#                hidden_valid, cur_data_valid, cur_targets_valid,
#                optimizer,
#                args.unrolled)

#        optimizer.zero_grad()
        # hidden[s_id] = repackage_hidden(hidden[s_id])
#        hidden = model.init_hidden(args.batch_size)


#        log_prob, hidden, rnn_hs, dropped_rnn_hs, probs = parallel_model(cur_data, hidden, return_h=True)
      
#        criterion = nn.CrossEntropyLoss()

#        raw_loss = criterion(log_prob, cur_targets)

#        loss = raw_loss
       
        # Temporal Activation Regularization (slowness)
#        loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
#        total_loss += raw_loss.data
#        loss.backward()
      

#        gc.collect()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
#        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
#        optimizer.step()
        
#        prec1,prec2 = utils.accuracy(log_prob, cur_targets, topk=(1,2)) 
        
#        n = input.size(0)
        
#        objs.update(loss.data, n)
       
#        top1.update(prec1.data, n) 

        
#        if step % report_freq == 0 and step > 0:
        
#            logging.info('| step {:3d} | train obj {:5.2f} | '
#                'train acc {:8.2f}'.format(step,
#                                           objs.avg, top1.avg))
          

#    logging.info('{} epoch done   '.format(epoch) + str(parallel_model.genotype()))

#    return top1.avg, objs.avg


#def evaluate(data_source, batch_size=10):
#    objs = utils.AvgrageMeter()
#    top1 = utils.AvgrageMeter()
#    top5 = utils.AvgrageMeter()
  
#    model.eval()
#    total_loss = 0
#    hidden = model.init_hidden(batch_size)
    
#    for step, (input, target) in enumerate(valid_data):

#        data, targets = input, target
        
#        targets = torch.max(targets, 1)[1]


#        log_prob, hidden = parallel_model(data, hidden)
#        criterion = nn.CrossEntropyLoss()

#        loss = criterion(log_prob, targets)
        
#        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 2))
#        n = input.size(0)
        
#        objs.update(loss.data, n)
#        top1.update(prec1.data, n)
#        top2.update(prec5.data, n)
        
#        if step % args.log_interval == 0:
#            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top2.avg)
#        hidden = repackage_hidden(hidden)
        
#    return top1.avg, objs.avg
    
  

if __name__ == '__main__':
    start_time = time.time()
    main() 
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Total searching time: %ds', duration)
