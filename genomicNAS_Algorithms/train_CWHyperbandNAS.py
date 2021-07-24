#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 09:53:23 2021

@author: amadeu
"""



# from generate_samples import generate_random_architectures
from randomSearch_and_Hyperband_Tools.random_Sampler import generate_random_architectures
#from transform_genotype import transform_Genotype
from randomSearch_and_Hyperband_Tools.utils import mask2geno, merge

# von DARTS
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
import copy

import gc

# original nur RNN version
import generalNAS_tools.genotypes

from randomSearch_and_Hyperband_Tools.model_search import RNNModelSearch

# import model_search as one_shot_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from generalNAS_tools.utils import repackage_hidden, create_exp_dir, save_checkpoint

import generalNAS_tools.utils

from randomSearch_and_Hyperband_Tools.hyperbandSampler import create_cnn_supersubnet, create_rhn_supersubnet

from randomSearch_and_Hyperband_Tools.train_and_validate import train, evaluate_architecture


#from ..data_preprocessing import get_data
import generalNAS_tools.data_preprocessing_new as dp


parser = argparse.ArgumentParser(description='DARTS for genomic Data')
#parser.add_argument('--data', type=str, default='/home/amadeu/anaconda3/envs/darts_env/cnn/data2/trainset.txt', help='location of the data corpus')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--cnn_lr', type=float, default=0.025, help='learning rate for CNN part')
parser.add_argument('--cnn_weight_decay', type=float, default=3e-4, help='weight decay for CNN part')
parser.add_argument('--rhn_lr', type=float, default=2, help='learning rate for CNN part')
parser.add_argument('--rhn_weight_decay', type=float, default=5e-7, help='weight decay for RHN part')

parser.add_argument('--num_disc', type=int, default=2, help='number of operations which are discarded after budget')
parser.add_argument('--budget', type=int, default=2, help='budget/epochs after operations are discarded')
parser.add_argument('--num_super_subnets', type=int, default=5, help='number of supersubnets which are created after ')

parser.add_argument('--validation', type=bool, default=True)
parser.add_argument('--report_validation', type=int, default=2, help='validation epochs') 

parser.add_argument('--num_steps', type=int, default=3, help='number of iterations per epoch')
parser.add_argument('--train_directory', type=str, default='/home/amadeu/Downloads/genomicData/train', help='directory of training data')
parser.add_argument('--valid_directory', type=str, default='/home/amadeu/Downloads/genomicData/validation', help='directory of validation data')
parser.add_argument('--num_files', type=int, default=3, help='number of files for training data')
parser.add_argument('--seq_len', type=int, default=200, help='sequence length')
parser.add_argument('--num_samples', type=int, default=3, help='number of random sampled architectures')
parser.add_argument('--next_character_prediction', default=True, action='store_true', help='task of model')

parser.add_argument('--one_clip', default=False, action='store_false', help='only one clip params or more')
parser.add_argument('--clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--conv_clip', type=float, default=0.25, help='gradient clipping of convs')
parser.add_argument('--rhn_clip', type=float, default=0.25, help='gradient clipping of lstms')

parser.add_argument('--init_channels', type=int, default=8, help='num of init channels')
parser.add_argument('--layers', type=int, default=3, help='total number of layers')
parser.add_argument('--num_classes', type=int, default=4, help='num of output classes') 
parser.add_argument('--steps', type=int, default=4, help='total number of Nodes')
parser.add_argument('--multiplier', type=int, default=4, help='multiplier')
parser.add_argument('--stem_multiplier', type=int, default=3, help='stem multiplier')

parser.add_argument('--epochs', type=int, default=2,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                    help='batch size')
#parser.add_argument('--dropout', type=float, default=0.75,
#                    help='dropout applied to layers (0 = no dropout)')
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
parser.add_argument('--report_freq', type=int, default=1, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='EXP',
                    help='path to save the final model')
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
args = parser.parse_args()


args.save = '{}search-{}-{}'.format(args.save, args.note, time.strftime("%Y%m%d-%H%M%S"))

create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
# logging.getLogger().addHandler(fh)
logging = logging.getLogger(__name__)


def main():
    
    train_start = time.strftime("%Y%m%d-%H%M")
    
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
        
    #random_architectures = generate_random_architectures(generate_num=args.num_samples) # generiert glaube ich 10 random adj. matritzen 
    
    #genotypes = [transform_Genotype(random_architecture['adjacency_matrix'], random_architecture['operations']) for random_architecture in
    #                         random_architectures]
                
    #genohashs = [hash(str(genotype)) for genotype in genotypes]
    
    
    #genotypes = [eval(arch) if isinstance(arch, str) else arch for arch in genotypes] # speichert die genotypes jetzt als liste 
    
    #### wenn randomSearch ohne WS, benutze subnet_masks ####
    #subnet_masks = [geno2mask(genotype) for genotype in genotypes] # transform genotypes into alpha matrix
  
            
    #### wenn randamsearchWS, benutze supernet_mask ####

    sub_masks = generate_random_architectures(2)

    cnn_masks = []
    rnn_masks = []
    for cnn_sub in sub_masks:
        cnn_masks.append(cnn_sub[0])
    for rnn_sub in sub_masks:
        rnn_masks.append(rnn_sub[1])
            
    supernet_mask = merge(cnn_masks, rnn_masks)
    supernet_mask = list(supernet_mask)
    #supernet_mask = merge(cnn_masks, rnn_masks) # die 5 subnets zusammengefügt, damit er 1 großes supernet hat, welches aus den 100 init_samples/subarchitecturen gebildet wurde
        
           
    # create supernet S         
    super_model = RNNModelSearch(args.seq_len, args.dropouth, args.dropoutx,
                            args.init_channels, args.num_classes, args.layers, args.steps, args.multiplier,
                            args.stem_multiplier, True, 0.2, None, supernet_mask)
        
    conv = []
    rhn = []
    for name, param in super_model.named_parameters():
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
    
    
    train_losses = []
    valid_losses = []
    acc_train = []
    acc_valid = []

    # pretrain a supernet S for some epochs, with all sampled edges, nodes and operations
    for epoch in range(args.budget): 
        # epoch=0
        # supernet.drop_path_prob = args.drop_path_prob * epoch / self.epochs
        # supernet.drop_path_prob = drop_path_prob * epoch / epochs
        lr = scheduler.get_last_lr()[0]
        
        train_start = time.strftime("%Y%m%d-%H%M")
   
        lr = scheduler.get_last_lr()[0]
        logging.info('Epoch: %d lr: %e', epoch, lr)
        epoch_start = time.time()

        
        train_acc, train_obj = train(train_object, super_model, criterion, optimizer, lr, epoch, rhn, conv, args.num_steps, clip_params, args.report_freq, args.beta, args.one_clip)

                                              
        scheduler.step()
        
        if args.validation == True:
               if epoch % args.report_validation == 0:
                   valid_acc, valid_obj = evaluate_architecture(valid_object, super_model, criterion, optimizer, epoch, args.report_freq, args.num_steps) 

                   logging.info('Valid_acc %f', valid_acc)
                    
                   valid_losses.append(valid_obj)
                   acc_valid.append(valid_acc)
                   
         train_losses.append(train_obj)
         acc_train.append(train_acc)
            
         logging.info('Train_acc %f', train_acc)
         epoch_duration = time.time() - epoch_start
         logging.info('Epoch time: %ds', epoch_duration)
            
        #if epoch == args.epochs-1:
        #    valid_acc, valid_obj = evaluate_architecture(valid_object, random_model, criterion, optimizer, epoch) 
            
        #    random_search_results.append((genotype, valid_acc)) # wenn wir z.B. 20 random samples haben, dann hätten wir liste mit 20 elementen
    
    # just for initialization
    #disc_ops_cnn = []
    #disc_ops_rhn = []
    supernet_mask = copy.deepcopy(supernet_mask)
    
    # supernet_mask[0]

    
    while (len(disc_ops_cnn) | len(disc_ops_rhn) != 0):
        
        # create the supersubnets and evaluate them
        # cnn_test = supernet_mask[0]
        super_model_weights = super_model.state_dict()   
        
        hyperband_results = []
        
        for i in range(args.num_super_subnets):
           mask_cnn, disc_ops_cnn = create_cnn_supersubnet(supernet_mask, args.num_disc)
           mask_rhn, disc_ops_rhn = create_rhn_supersubnet(supernet_mask, args.num_disc)
           supersubnet_mask = [mask_cnn, mask_rhn]
           # 
                      
           supersubnet_model = RNNModelSearch(args.seq_len, args.dropouth, args.dropoutx,
                                args.init_channels, args.num_classes, args.layers, args.steps, args.multiplier,
                                args.stem_multiplier, True, 0.2, None, supersubnet_mask)
           
           supersubnet_weights = supersubnet_model.state_dict()
           
           trained_weights = {k: v for k, v in super_model_weights.items() if k in supersubnet_weights}
           #trained_weights = {k: v for k, v in supersubnet_weights.items() if k in super_model_weights}

           # müssen die old weights sein

           supersubnet_weights.update(trained_weights)
           
           supersubnet_model.load_state_dict(supersubnet_weights) 

           # validate the supersubnets using weights from pretrained supernet S
           for epoch in range(1):               
               valid_acc, valid_obj = evaluate_architecture(valid_object, supersubnet_model, criterion, optimizer, epoch, args.report_freq, args.num_steps) 

               # valid_acc, valid_obj = evaluate_architecture(valid_object, supersubnet_model, criterion, optimizer, epoch) 
               hyperband_results.append([supersubnet_mask, supersubnet_model, valid_acc]) # hätten am Ende z.B. 5 hyperband_results
                
        def acc_position(list):
            return list[2]
    
        hyperband_results.sort(reverse=True, key=acc_position) # in descending order: best/highest acc is on top (contains the worst operations which we want to discard)
        
        # discard the edges and operations from the good performing supersubnets to build new S
        # keep mask with best/highest acc 
        supernet_mask = hyperband_results[0][0]
        super_model = hyperband_results[0][1]
        
        #super_model = RNNModelSearch(args.seq_len, args.dropout, args.dropouth, args.dropoutx, args.dropouti, args.dropoute,
        #                        args.init_channels, args.num_classes, args.layers, args.steps, args.multiplier,
        #                        args.stem_multiplier, True, 0.2, None, supernet_mask)
        
        #super_model_weights = super_model.state_dict() # diese 969
        # jetzt muss ich diejenigen discarden, damit nur noch diese 935 da sind (aber die gewichte immer noch beibehalten)

        
        # now train the remaining supernet S for few epochs (according to budget)
        for epoch in range(args.budget): 
            # epoch=0
            lr = scheduler.get_last_lr()[0]
        
            train_start = time.strftime("%Y%m%d-%H%M")
   
            lr = scheduler.get_last_lr()[0]
            logging.info('Epoch: %d lr: %e', epoch, lr)
            epoch_start = time.time()

            # supernet.drop_path_prob = args.drop_path_prob * epoch / self.epochs
            # supernet.drop_path_prob = drop_path_prob * epoch / epochs
            train_acc, train_obj = train(train_object, super_model, criterion, optimizer, lr, epoch, rhn, conv, args.num_steps, clip_params, args.report_freq, args.beta, args.one_clip)

            # train_acc, train_obj = train(train_object, super_model, criterion, optimizer, epoch, rhn, conv)
                                      
            scheduler.step()
            
            if args.validation == True:
               if epoch % args.report_validation == 0:
                   valid_acc, valid_obj = evaluate_architecture(valid_object, super_model, criterion, optimizer, epoch, args.report_freq, args.num_steps) 

                   logging.info('Valid_acc %f', valid_acc)
                    
                   valid_losses.append(valid_obj)
                   acc_valid.append(valid_acc)
                   
         train_losses.append(train_obj)
         acc_train.append(train_acc)
            
         logging.info('Train_acc %f', train_acc)
         epoch_duration = time.time() - epoch_start
         logging.info('Epoch time: %ds', epoch_duration)
         
          
    # hyperband_results.sort(reverse=True, key=acc_position)
    # final/best architecture
    
    genotype = mask2geno(supernet_mask)
    
    print(genotype)

    genotype_file = '{}-hb_geno'.format(args.save)
    np.save(genotype_file, genotype)
    
    mask_file = '{}-hb_mask'.format(args.save)
    # np.save(hyperband_file, supernet_mask, genotype)
    np.save(mask_file, supernet_mask)
    
            
    trainloss_file = '{}-train_loss-{}'.format(args.save, train_start)
    np.save(trainloss_file, train_losses_all)
      
    acctrain_file = '{}-acc_train-{}'.format(args.save, train_start) 
    np.save(acctrain_file, acc_train_all)
         
    validloss_file = '{}-valid_loss-{}'.format(args.save, train_start)
    np.save(validloss_file, valid_losses_all)
      
    accvalid_file = '{}-acc_valid-{}'.format(args.save, train_start) 
    np.save(acctrain_file, acc_valid_all)

  

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