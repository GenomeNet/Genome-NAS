#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 21:46:23 2021

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

import gc

import torch.utils

# original nur RNN version
import generalNAS_tools.genotypes
# from randomSearch_and_Hyperband_Tools.model_search import RNNModelSearch

import model_search as one_shot_model

from randomSearch_and_Hyperband_Tools.random_Sampler import generate_random_architectures, mask2genotype
# from transform_genotype import transform_Genotype

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from randomSearch_and_Hyperband_Tools.utils import mask2geno, geno2mask, merge

from generalNAS_tools.utils import repackage_hidden, create_exp_dir, save_checkpoint

from generalNAS_tools import utils

#from ..data_preprocessing import get_data

# from randomSearch_and_Hyperband_Tools.train_and_validate import train, evaluate_architecture
from generalNAS_tools.train_and_validate import train, infer

from randomSearch_and_Hyperband_Tools.hb_iteration import hb_step

from generalNAS_tools.utils import scores_perClass, scores_Overall, pr_aucPerClass, roc_aucPerClass, overall_acc, overall_f1



parser = argparse.ArgumentParser(description='DARTS for genomic Data')
parser.add_argument('--data', type=str, default='/home/amadeu/anaconda3/envs/darts_env/cnn/data2/trainset.txt', help='location of the data corpus')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--cnn_lr', type=float, default=0.025, help='learning rate for CNN part')
parser.add_argument('--cnn_weight_decay', type=float, default=3e-4, help='weight decay for CNN part')
parser.add_argument('--rhn_lr', type=float, default=2, help='learning rate for CNN part')
parser.add_argument('--rhn_weight_decay', type=float, default=5e-7, help='weight decay for RHN part')

parser.add_argument('--num_steps', type=int, default=2, help='number of iterations per epoch')
parser.add_argument('--iterations', type=int, default=3, help='number of iterations per epoch')

parser.add_argument('--train_directory', type=str, default='/home/amadeu/Downloads/genomicData/train', help='directory of training data')
parser.add_argument('--valid_directory', type=str, default='/home/amadeu/Downloads/genomicData/validation', help='directory of validation data')
parser.add_argument('--test_directory', type=str, default='/home/amadeu/Downloads/genomicData/test', help='directory of test data')

parser.add_argument('--train_input_directory', type=str, default='/home/amadeu/Desktop/GenomNet_MA/data/inputs_small.pkl', help='directory of train data')
parser.add_argument('--train_target_directory', type=str, default='/home/amadeu/Desktop/GenomNet_MA/data/targets_small.pkl', help='directory of train data')
parser.add_argument('--valid_input_directory', type=str, default='/home/amadeu/Desktop/GenomNet_MA/data/inputs_small_val.pkl', help='directory of validation data')
parser.add_argument('--valid_target_directory', type=str, default='/home/amadeu/Desktop/GenomNet_MA/data/targets_small_val.pkl', help='directory of validation data')
parser.add_argument('--test_input_directory', type=str, default='/home/amadeu/Desktop/GenomNet_MA/data/inputs_small_test.pkl', help='directory of test data')
parser.add_argument('--test_target_directory', type=str, default='/home/amadeu/Desktop/GenomNet_MA/data/targets_small_test.pkl', help='directory of test data')

parser.add_argument('--task', type=str, default='TF_bindings', help='defines the task')#TF_bindings

parser.add_argument('--validation', type=bool, default=True)
parser.add_argument('--report_validation', type=int, default=2, help='validation epochs') 

parser.add_argument('--next_character_prediction', type=bool, default=True, help='task of model')

parser.add_argument('--num_files', type=int, default=3, help='number of files for training data')
parser.add_argument('--seq_size', type=int, default=1000, help='sequence length') # 1000
parser.add_argument('--num_samples', type=int, default=2, help='number of random sampled architectures')

parser.add_argument('--one_clip', type=bool, default=True)
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--conv_clip', type=float, default=5, help='gradient clipping of convs')
parser.add_argument('--rhn_clip', type=float, default=0.25, help='gradient clipping of lstms')

parser.add_argument('--init_channels', type=int, default=8, help='num of init channels')
parser.add_argument('--layers', type=int, default=6, help='total number of layers')
parser.add_argument('--num_classes', type=int, default=4, help='num of output classes') 
parser.add_argument('--steps', type=int, default=4, help='total number of Nodes')
#parser.add_argument('--multiplier', type=int, default=4, help='multiplier')
#parser.add_argument('--stem_multiplier', type=int, default=3, help='stem multiplier')

parser.add_argument('--budget', type=int, default=2,
                    help='upper epoch limit')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.75,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.25,
                    help='dropout for hidden nodes in rnn layers (0 = no dropout)')
parser.add_argument('--dropoutx', type=float, default=0.75,
                    help='dropout for input nodes in rnn layers (0 = no dropout)')
#parser.add_argument('--dropouti', type=float, default=0.2,
#                    help='dropout for input embedding layers (0 = no dropout)')
#parser.add_argument('--dropoute', type=float, default=0,
#                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--seed', type=int, default=3,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--report_freq', type=int, default=1, metavar='N',
                    help='report interval')

parser.add_argument('--alpha', type=float, default=0,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1e-3,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--continue_train', action='store_true',
                    help='continue train from a checkpoint')
parser.add_argument('--max_seq_len_delta', type=int, default=20,
                    help='max sequence length')
parser.add_argument('--single_gpu', default=True, action='store_false', 
                    help='use single GPU')
parser.add_argument('--gpu', type=int, default=0, help='GPU device to use')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--note', type=str, default='try', help='note for this run')

parser.add_argument('--save', type=str,  default='search',
                    help='name to save the labels and predicitons')
parser.add_argument('--save_dir', type=str,  default= 'test_search',
                    help='path to save the labels and predicitons')
args = parser.parse_args()


utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
#logging.getLogger().addHandler(fh)

logging = logging.getLogger(__name__)


def main():
    
    torch.manual_seed(args.seed)
      
    logging.info("args = %s", args)
           
    #train_object, valid_object, num_classes = dp.data_preprocessing(train_directory = args.train_directory, valid_directory = args.valid_directory, num_files=args.num_files,
    #        seq_size = args.seq_len, batch_size=args.batch_size, next_character=args.next_character_prediction)
    
    #_, valid_data, num_classes = dp.data_preprocessing(train_directory = args.train_directory, valid_directory = args.valid_directory, num_files=args.num_files,
    #        seq_size = args.seq_len, batch_size=args.batch_size, next_character=args.next_character_prediction)
    
    if args.task == ("next_character_prediction" or "sequence_to_sequence"):
        
        import generalNAS_tools.data_preprocessing_new as dp

        train_queue, valid_queue, num_classes = dp.data_preprocessing(train_directory = args.train_directory, valid_directory = args.valid_directory, num_files=args.num_files,
                seq_size = args.seq_size, batch_size=args.batch_size, next_character=args.next_character_prediction)
      
        _, test_queue, _ = dp.data_preprocessing(train_directory = args.train_directory, valid_directory = args.test_directory, num_files=args.num_files,
                seq_size = args.seq_size, batch_size=args.batch_size, next_character=args.next_character_prediction)
        
        criterion = nn.CrossEntropyLoss().to(device)
        
    if args.task == 'TF_bindings':
        
        import generalNAS_tools.data_preprocessing_TF as dp
        
#        train_queue, valid_queue, test_queue = dp.data_preprocessing(args.train_input_directory, args.valid_input_directory, args.test_input_directory, args.train_target_directory, args.valid_target_directory, args.test_target_directory, args.batch_size)
        train_queue, valid_queue, test_queue = dp.data_preprocessing(args.train_directory, args.valid_directory, args.test_directory, args.batch_size)

        criterion = nn.BCELoss().to(device)

        num_classes = 919
           
        
    random_architectures = generate_random_architectures(generate_num=args.num_samples) # generiert glaube ich 10 random adj. matritzen 
    
    cnn_masks = []
    rnn_masks = []
    for cnn_sub in random_architectures:
        cnn_masks.append(cnn_sub[0])
    for rnn_sub in random_architectures:
        rnn_masks.append(rnn_sub[1])
            
    #### wenn randamsearchWS, benutze supernet_mask ####
    #supernet_mask = merge(cnn_masks, rnn_masks) # die 5 subnets zusammengefügt, damit er 1 großes supernet hat, welches aus den 100 init_samples/subarchitecturen gebildet wurde
    
    random_search_results = []
    
    count=0

    # store metrics of each configuration here
    train_losses_all = []
    valid_losses_all = []
    acc_train_all = []
    acc_valid_all = []
    
    time_per_config = []
    
    # initial step
    for mask in random_architectures: # iterate over all architectures
    
        configuration_start = time.time()

        # mask = random_architectures[1]
        count += 1
        
        genotype = mask2geno(mask)
        
        logging.info('| num_architecture {:3d}'.format(count))
           
        logging.info(genotype)   

        multiplier, stem_multiplier = 4,3
        
        random_model = one_shot_model.RNNModel(args.seq_size, args.dropouth, args.dropoutx,
                            args.init_channels, num_classes, args.layers, args.steps, multiplier, stem_multiplier,
                            False, 0.2, genotype=genotype, task=args.task).to(device)
        
        conv = []
        rhn = []
        for name, param in random_model.named_parameters():
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
            
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
        
        clip_params = [args.conv_clip, args.rhn_clip, args.clip]
        
        train_losses, valid_losses, train_acc, valid_acc, acc = hb_step(train_queue, valid_queue, random_model, rhn, conv, criterion, scheduler, args.batch_size, optimizer, None, None, args.unrolled, args.num_steps, clip_params, args.report_freq, args.beta, args.one_clip, args.task, args.budget)
                    
        random_search_results.append([mask, random_model, optimizer, scheduler, acc]) # wenn wir z.B. 20 random samples haben, dann hätten wir liste mit 20 elementen
          
        train_losses_all.append(train_losses)
        valid_losses_all.append(valid_losses)
        acc_train_all.append(train_acc)
        acc_valid_all.append(valid_acc)
        
       
    # check_opt1 = optimizer
    # check_opt1.param_groups
    # check_opt2 = optimizer
    # check_opt2.param_groups

         
    def acc_position(list):
        return list[4]
    
    random_search_results.sort(reverse=True, key=acc_position)
    
    num_keep_archs = len(random_search_results)//2
    
    random_architectures=[]
    for keep in range(num_keep_archs):
        # print(keep)
        random_architectures.append(random_search_results[keep])
        
    for iters in range(args.iterations):
        
        logging.info('| iters {:3d} | num_keep_archs {:5.2f}'.format(iters, num_keep_archs))

        random_search_results = []
        
        for archs in enumerate(random_architectures):

            # archs = random_architectures[0]
            mask = archs[1][0]
            random_model = archs[1][1]
            optimizer = archs[1][2]
            scheduler = archs[1][3]
            
            conv = []
            rhn = []
            for name, param in random_model.named_parameters():
                #print(name)
                #if 'stem' or 'preprocess' or 'conv' or 'bn' or 'fc' in name:
                if 'rnns' in name:
                    #print(name)
                    rhn.append(param)
                #elif 'decoder' in name:
                else:
                    #print(name)
                    conv.append(param)
                    
            train_losses, valid_losses, train_acc, valid_acc, acc = hb_step(train_queue, valid_queue, random_model, rhn, conv, criterion, scheduler, args.batch_size, optimizer, None, None, args.unrolled, args.num_steps, clip_params, args.report_freq, args.beta, args.one_clip, args.task, args.budget)

            random_search_results.append([mask, random_model, optimizer, scheduler, acc]) # wenn wir z.B. 20 random samples haben, dann hätten wir liste mit 20 elementen
            train_losses_all.append(train_losses)
            valid_losses_all.append(valid_losses)
            acc_train_all.append(train_acc)
            acc_valid_all.append(valid_acc)
            
        def acc_position(list):
            return list[4]

        random_search_results.sort(reverse=True, key=acc_position)

        num_keep_archs = len(random_search_results)//2

        random_architectures=[]
        for keep in range(num_keep_archs):
            # print(keep)
            random_architectures.append(random_search_results[keep])
        
        
    
    # final/best architecture
    final_mask = random_search_results[0][0]
    
    genotype = mask2geno(final_mask)

    # random_search_file = 'randomSearchResults-{}'.format(args.save)
    # np.save(os.path.join(args.save_dir, random_search_file), random_search_results)
    
    genotype_file = 'random_geno-{}'.format(args.save)
    np.save(os.path.join(args.save_dir, genotype_file), genotype)
    
    trainloss_file = 'train_loss-{}'.format(args.save)
    np.save(os.path.join(args.save_dir, trainloss_file), train_losses_all)
    
    acc_train_file = 'acc_train-{}'.format(args.save)
    np.save(os.path.join(args.save_dir, acc_train_file), acc_train_all)

    #time_file = 'time-{}'.format(args.save)
    #np.save(os.path.join(args.save_dir, time_file), time_per_config)

    # safe valid data
    validloss_file = 'valid_loss-{}'.format(args.save)
    np.save(os.path.join(args.save_dir, validloss_file), valid_losses_all)

    acc_valid_file = 'acc_valid-{}'.format(args.save)
    np.save(os.path.join(args.save_dir, acc_valid_file), acc_valid_all)
    

# abchecken wie ob er auch wirklich die gewichte übernimmt (indem ich für 5 stück genau in ablage den f1-score mit erstem gewicht abspeichere)
# files am ende
# jede iteration printen lassen
# mask muss noch jedes mal neu, optimizer muss output werden


if __name__ == '__main__':
    start_time = time.time()
    main() 
    end_time = time.time()
    duration = end_time - start_time
    duration_m = duration/60
    duration_h = duration_m/60
    duration_d = duration_h/24
    logging.info('Total searching time: %ds', duration)
    logging.info('Total searching time: %dd', duration_d)
