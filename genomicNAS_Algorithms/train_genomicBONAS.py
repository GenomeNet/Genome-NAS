#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 09:23:20 2021

@author: amadeu
"""


import logging
import os
from BO_tools.configure_files import local_root_dir, local_data_dir, logfile, taskname, results_dir
from BO_tools.runner import Runner
import argparse

import os



parser = argparse.ArgumentParser("search")
parser.add_argument('--gpu', type=str, default='0',
                    help='gpu')
parser.add_argument('--train_directory', type=str, default='/home/amadeu/Downloads/genomicData/train', help='directory of training data')
parser.add_argument('--valid_directory', type=str, default='/home/amadeu/Downloads/genomicData/validation', help='directory of validation data')
parser.add_argument('--num_files', type=int, default=3, help='number of files for data')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--cnn_lr', type=float, default=0.025, help='learning rate for CNN part')
parser.add_argument('--cnn_weight_decay', type=float, default=3e-4, help='weight decay for CNN part')
parser.add_argument('--rhn_lr', type=float, default=2, help='learning rate for RHN part')
parser.add_argument('--rhn_weight_decay', type=float, default=5e-7, help='weight decay for RHN part')
parser.add_argument('--workers', type=int, default=2, help='number of workers to load dataset')
parser.add_argument('--num_steps', type=int, default=2, help='number of iterations per epoch')
parser.add_argument('--next_character_prediction', type=bool, default=True, help='task of model')
parser.add_argument('--one_clip', type=bool, default=True)
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--conv_clip', type=float, default=5, help='gradient clipping of convs')
parser.add_argument('--rhn_clip', type=float, default=0.25, help='gradient clipping of lstms')
parser.add_argument('--seq_len', type=int, default=200, help='sequence length')

parser.add_argument('--gcn_epochs', type=int, default=10, help='epochs of GCN surrogate')
parser.add_argument('--gcn_lr', type=float, default=0.001, help='learning rate for GCN surrogate')
parser.add_argument('--loss_num', type=int, default=3, help='number of losses')
parser.add_argument('--generate_num', type=int, default=5, help='number of sampled child models in each BONAS iteration')
parser.add_argument('--iterations', type=int, default=5, help='iterations of BONAS algorithm')
parser.add_argument('--bo_sample_num', type=int, default=5, help='number of child models, which are run in each BONAS iteration')
parser.add_argument('--sample_method', type=str, default='random', help='sample method for child models')
parser.add_argument('--if_init_samples', type=bool, default=True)
parser.add_argument('--init_num', type=int, default=5, help='number of child models for initial design')

parser.add_argument('--dropouth', type=float, default=0.25,
                    help='dropout for hidden nodes in rnn layers (0 = no dropout)')
parser.add_argument('--dropoutx', type=float, default=0.75,
                    help='dropout for input nodes in rnn layers (0 = no dropout)')

parser.add_argument('--train_supernet_epochs', type=int, default=1, help='epochs of supernet')
parser.add_argument('--super_batch_size', type=int, default=2, help='batch_size for supernet')
parser.add_argument('--sub_batch_size', type=int, default=2, help='batch_size for subnets')

parser.add_argument('--report_freq', type=int, default=1000, help='report frequency')
parser.add_argument('--epochs', type=int, default=15, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=8, help='num of init channels')
parser.add_argument('--layers', type=int, default=3, help='total number of layers')

parser.add_argument('--drop_path_prob', type=float, default=0.2, help='report frequency')
parser.add_argument('--mode', type=str, default='random', help='sample method for child models')

#parser.add_argument('--taskname', type=str, default='bonas_model', help='name of task')
#parser.add_argument('--local_root_dir', type=str, default='/home/amadeu/', help='directory, where results are saved')
#parser.add_argument('--local_data_dir', type=str, default='/home/amadeu/data_BONAS', help='directory, where data ist stored')
#parser.add_argument('--results_dir', type=str, default='trained_results', help='directory, where results are saved')
args = parser.parse_args()



os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
logging.basicConfig(filename=os.path.join(local_root_dir, results_dir, taskname, str(args.gpu)+logfile), filemode='w', level=logging.INFO,
                    format='%(asctime)s : %(levelname)s  %(message)s', datefmt='%Y-%m-%d %A %H:%M:%S')
os.environ["PYTHONHASHSEED"] = "0"
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s : %(levelname)s  %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)





search_config = dict(
    gcn_epochs=args.gcn_epochs,#100,
    gcn_lr=args.gcn_lr,
    loss_num=args.loss_num,
    generate_num=args.generate_num,#100,
    iterations=args.iterations,
    bo_sample_num=args.bo_sample_num,#100,
    sample_method=args.sample_method,
    if_init_samples=args.if_init_samples,
    init_num=args.init_num,#100,
)




training_config = dict(
    train_supernet_epochs=args.train_supernet_epochs,
    data_path=os.path.join(local_data_dir, 'data'),
    super_batch_size=args.super_batch_size,#64,
    sub_batch_size=args.sub_batch_size,#128,
    
    cnn_lr=args.cnn_lr,
    cnn_weight_decay=args.cnn_weight_decay,
    rhn_lr=args.rhn_lr,
    rhn_weight_decay=args.rhn_weight_decay,
    momentum=args.momentum,
    
    report_freq=args.report_freq,
    epochs=args.epochs,
    init_channels=args.init_channels,#36,
    layers= args.layers,#20,
    drop_path_prob=args.drop_path_prob,
    seed=0,
    #grad_clip=args.clip,
    parallel=False,
    mode=args.mode,
    train_directory = args.train_directory,
    valid_directory = args.valid_directory,
    num_files = args.num_files,
    seq_len = args.seq_len,
    num_steps = args.num_steps,
    next_character_prediction=args.next_character_prediction,
    dropouth = args.dropouth,
    dropoutx = args.dropoutx,
    one_clip = args.one_clip,
    clip = args.clip,
    conv_clip = args.conv_clip,
    rhn_clip = args.rhn_clip 
)





if __name__ == "__main__":
    runner = Runner(**search_config, training_cfg=training_config) # training_config und search_config wird beides von
    runner.run()