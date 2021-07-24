#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 17:12:53 2021

@author: amadeu
"""

from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat rnn rnn_concat')


PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'tanh',
    'identity',
    'sigmoid',
    'relu'
]

PRIMITIVES_cnn = [
    'none',
    'max_pool_5',
    'avg_pool_5',
    'skip_connect',
    'sep_conv_15',
    'sep_conv_9',
    'dil_conv_15',
    'dil_conv_9',
    'conv_15'
]


PRIMITIVES_rnn = [
    'none',
    'tanh',
    'identity',
    'sigmoid',
    'relu'
]

rnn_steps = 8
CONCAT = 8




OPS_cnn = ['input_cnn', 'max_pool_5', 'skip_connect', 'sep_conv_14', 'dil_conv_14']
# for my RHN I allow "none" operation
OPS_rnn = ['tanh', 'identity', 'sigmoid', 'relu', 'output_rnn']

