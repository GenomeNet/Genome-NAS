#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 20:47:00 2021

@author: amadeu
"""


from generalNAS_tools.genotypes import PRIMITIVES_cnn, PRIMITIVES_rnn, rnn_steps, CONCAT, Genotype
import numpy as np

import copy

import torch
import torch.nn as nn
import os, shutil
from torch.autograd import Variable



def mask2geno(mask):
    cnn_mask = mask[0]
    rhn_mask = mask[1]
      
    def _parse_cnn(cnn_mask):
        gene_cnn = []
        n = 2
        start = 0
        for i in range(4):
            # i=3
                
            end = start + n 
            m = cnn_mask[start:end].copy() # die ersten beiden Zeilen von alphas
              
            ## hier bestimmt er welche verbindung/Input von den 2 möglichen bei i=0, bzw. 3 möglichen bei i=1, bzw. 4 möglichen bei i=2
            edges, ops = np.nonzero(m)
            
            for j,h in zip(ops,edges): #
                gene_cnn.append((PRIMITIVES_cnn[j], h))
            
            start = end
            n += 1
        return gene_cnn
            
    def _parse_rhn(rhn_mask):
        gene_rhn = []
        n = 1
        start = 0
        for i in range(8):
            # i=0
            end = start + n 
            m = rhn_mask[start:end].copy() # die ersten beiden Zeilen von alphas
              
            ## hier bestimmt er welche verbindung/Input von den 2 möglichen bei i=0, bzw. 3 möglichen bei i=1, bzw. 4 möglichen bei i=2
            edges, ops = np.nonzero(m)
            
            for j, h in zip(ops, edges): #
                gene_rhn.append((PRIMITIVES_rnn[j], h))
            
            start = end
            n += 1
        return gene_rhn
            
    gene_normal = _parse_cnn(cnn_mask)            
    gene_reduce = _parse_cnn(cnn_mask)
      
    gene_rnn = _parse_rhn(rhn_mask)
        
    concat = range(2+4-4, 4+2)
     

    genotype = Genotype(normal=gene_normal, normal_concat=concat, reduce=gene_reduce, reduce_concat=concat, rnn=gene_rnn, rnn_concat=range(8+1)[-CONCAT:])
      
    return genotype


def geno2mask(genotype):
    des = -1
    #mask = np.zeros([14+36,8+4])
    
    cnn_mask = np.zeros([14,8])
    rnn_mask = np.zeros([36,5])

    op_names_cnn, indices = zip(*genotype.normal)
    for cnt, (name, index) in enumerate(zip(op_names_cnn, indices)):
     
        if cnt % 2 == 0: 
       
            des += 1 
            total_state = sum(i + 2 for i in range(des)) 
           
        op_idx = PRIMITIVES_cnn.index(name) 
        node_idx = index + total_state 
        
        cnn_mask[node_idx, op_idx] = 1 
      
    op_names_rnn, indices = zip(*genotype.rnn)

    aux = 0
    for cnt, (name, index) in enumerate(zip(op_names_rnn, indices)):
   
       
        node_idx = aux + index
      
        op_idx = PRIMITIVES_rnn.index(name) 
    
        rnn_mask[node_idx, op_idx] = 1 
        
        aux += cnt +1
        
    return cnn_mask, rnn_mask

#subnet_mask = geno2mask(genotype)

def merge(cnn_mask, rnn_mask):
    supernet_mask_cnn = np.zeros((14, 9))
    supernet_mask_rnn = np.zeros((36,5))
    for mask in cnn_mask: 
        supernet_mask_cnn = mask + supernet_mask_cnn
        
    for mask in rnn_mask: 
        supernet_mask_rnn = mask + supernet_mask_rnn
    return supernet_mask_cnn, supernet_mask_rnn