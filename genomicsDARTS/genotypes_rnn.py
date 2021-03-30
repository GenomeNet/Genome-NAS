#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 10:28:20 2021

@author: amadeu
"""


from collections import namedtuple

total_genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat recurrent concat')

PRIMITIVES = [
    'none',
    'tanh',
    'relu',
    'sigmoid',
    'identity'
]
STEPS = 8
CONCAT = 8

#ENAS = Genotype(
#    recurrent = [
#        ('tanh', 0),
#        ('tanh', 1),
#        ('relu', 1),
#        ('tanh', 3),
#        ('tanh', 3),
#        ('relu', 3),
#        ('relu', 4),
#        ('relu', 7),
#        ('relu', 8),
#        ('relu', 8),
#        ('relu', 8),
#    ],
#    concat = [2, 5, 6, 9, 10, 11]
#)

#DARTS_V1 = Genotype(recurrent=[('relu', 0), ('relu', 1), ('tanh', 2), ('relu', 3), ('relu', 4), ('identity', 1), ('relu', 5), ('relu', 1)], concat=range(1, 9))
#DARTS_V2 = Genotype(recurrent=[('sigmoid', 0), ('relu', 1), ('relu', 1), ('identity', 1), ('tanh', 2), ('sigmoid', 5), ('tanh', 3), ('relu', 5)], concat=range(1, 9))
DARTS_V3 = total_genotype(normal=[('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 0), ('dil_conv_3x3', 2), ('max_pool_3x3', 0), ('skip_connect', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('dil_conv_5x5', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 3), ('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6), recurrent=[('relu', 0), ('identity', 0), ('sigmoid', 2), ('sigmoid', 2), ('identity', 3), ('relu', 2), ('sigmoid', 3), ('relu', 5)], concat=range(1, 9))

DARTS = DARTS_V3

#s1_1 = Genotype(recurrent=[('relu', 0), ('sigmoid', 1), ('relu', 2), ('tanh', 0), ('identity', 4), ('sigmoid', 4), ('sigmoid', 5), ('tanh', 3)], concat=range(1, 9))
#s1_2 = Genotype(recurrent=[('tanh', 0), ('tanh', 0), ('identity', 0), ('sigmoid', 2), ('identity', 4), ('identity', 5), ('identity', 6), ('identity', 7)], concat=range(1, 9))
#s1_3 = Genotype(recurrent=[('tanh', 0), ('tanh', 0), ('identity', 2), ('sigmoid', 2), ('identity', 4), ('identity', 5), ('identity', 6), ('identity', 7)], concat=range(1, 9))

#s2_1 = Genotype(recurrent=[('relu', 0), ('relu', 1), ('sigmoid', 2), ('identity', 1), ('relu', 2), ('relu', 1), ('identity', 2), ('tanh', 3)], concat=range(1, 9))
#s2_2 = Genotype(recurrent=[('relu', 0), ('relu', 1), ('identity', 1), ('identity', 1), ('relu', 2), ('identity', 1), ('tanh', 4), ('identity', 4)], concat=range(1, 9))
# s2_3 = Genotype(recurrent=[('relu', 0), ('relu', 1), ('identity', 1), ('identity', 1), ('relu', 2), ('identity', 1), ('tanh', 4), ('identity', 4)], concat=range(1, 9))


#s3_1 = Genotype(recurrent=[('identity', 0), ('relu', 1), ('identity', 0), ('identity', 2), ('sigmoid', 4), ('tanh', 0), ('sigmoid', 1), ('identity', 6)], concat=range(1, 9))
#s3_2 = Genotype(recurrent=[('relu', 0), ('identity', 1), ('identity', 0), ('identity', 3), ('identity', 0), ('identity', 4), ('identity', 1), ('relu', 4)], concat=range(1, 9))
#s3_3 = Genotype(recurrent=[('relu', 0), ('relu', 1), ('identity', 0), ('identity', 0), ('tanh', 1), ('identity', 1), ('identity', 1), ('relu', 4)], concat=range(1, 9))


#s4_1 = Genotype(recurrent=[('sigmoid', 0), ('tanh', 0), ('relu', 0), ('tanh', 2), ('sigmoid', 2), ('relu', 2), ('tanh', 4), ('relu', 0)], concat=range(1, 9))
#s4_2 = Genotype(recurrent=[('identity', 0), ('identity', 1), ('relu', 2), ('identity', 3), ('relu', 3), ('identity', 3), ('identity', 0), ('identity', 3)], concat=range(1, 9))
#s4_3 = Genotype(recurrent=[('identity', 0), ('identity', 1), ('relu', 2), ('tanh', 0), ('relu', 3), ('identity', 3), ('identity', 0), ('identity', 0)], concat=range(1, 9))