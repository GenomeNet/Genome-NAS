#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 10:47:48 2021

@author: amadeu
"""

import os



taskname = "supermodel_random_100"
local_root_dir = "/home/amadeu/" # root working directory "/home/amadeu/anaconda3/envs/GenomNet_MA/BONAS" #
local_data_dir = "/home/amadeu/data_BONAS" # data root "/home/amadeu/anaconda3/envs/GenomNet_MA/BONAS/data"
results_dir = "trained_results"
trained_pickle_file = "trained_models.pkl"
trained_csv_file = "trained_models.csv"
logfile = 'BOGCN_open_domain.log'
    
io_config = dict(
     trained_pickle_file=os.path.join(local_root_dir, results_dir, taskname, trained_pickle_file),
     trained_csv_file=os.path.join(local_root_dir, results_dir, taskname, trained_csv_file),
)
    
distributed = False
