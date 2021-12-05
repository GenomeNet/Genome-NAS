#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 10:14:21 2021

@author: amadeu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from generalNAS_tools.operations_14_9 import *
from generalNAS_tools.genotypes import OPS_cnn, OPS_rnn, PRIMITIVES_cnn, PRIMITIVES_rnn, rnn_steps, Genotype, CONCAT

# from genotypes import PRIMITIVES_cnn, rnn_steps, PRIMITIVES_rnn, CONCAT
import generalNAS_tools.genotypes

from torch.autograd import Variable
from collections import namedtuple
from randomSearch_and_Hyperband_Tools.model import DARTSCell, RNNModel

from darts_tools.comp_aux import compute_positions, activ_fun

import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#import cnn_eval

# for search

# ninp, nhid, dropouth, dropoutx, mask
class DARTSCellSearch(DARTSCell):

  def __init__(self, ninp, nhid, dropouth, dropoutx, switch_rnn):
    super(DARTSCellSearch, self).__init__(ninp, nhid, dropouth, dropoutx, genotype=None)
    
    self.bn = nn.BatchNorm1d(nhid, affine=False)
    # self.mask_rnn = mask_rnn
    self.switch_rnn = switch_rnn
    
    self.rows, self.nodes, self.cols = compute_positions(self.switch_rnn, rnn_steps, PRIMITIVES_rnn)
    # switch_rnn = switches_rnn
    # rows, nodes, cols = compute_positions(switch_rnn, rnn_steps, PRIMITIVES_rnn)

    self.acn, self.acf = activ_fun(rnn_steps, PRIMITIVES_rnn, self.switch_rnn)  
    # acn, acf = activ_fun(rnn_steps, PRIMITIVES_rnn, switch_rnn)  

  def cell(self, x, h_prev, x_mask, h_mask):
      
    s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)
    #print(s0.shape)
    s0 = self.bn(s0)
    states = s0.unsqueeze(0) # [1,2,256]
    offset=0
    for i in range(rnn_steps):
      # i = 3
      if self.training:
        masked_states = states * h_mask.unsqueeze(0) 
      else:
        masked_states = states # bei i=0 [1,2,256], bei i=1 [2,2,256]
        
      ch = masked_states.view(-1, self.nhid).mm(self._Ws[i]).view(i+1, -1, 2*self.nhid) # [1,2,256] wird mit dem i-ten element von _Ws multipliziert also die 8 elmentige weight  [256, 512] geht er durch 
      # ch = masked_states.view(-1, nhid).mm(_Ws[i]).view(i+1, -1, 2*nhid) 

      c, h = torch.split(ch, self.nhid, dim=-1)
      
      c = c.sigmoid()
    
      s = torch.zeros_like(s0) # [2,256] nur mit 0en bei jedem i !
     
      for k, name in enumerate(self.acn[i]):
      
        fn = self._get_activation(name)
     
        unweighted = states + c * (fn(h) - states) # states [3,2,256], wobei s immer [2,256]
     
        s += torch.sum(unweighted[self.nodes[offset], :, :], dim=0) 
        offset+=1

      s = self.bn(s) 
      states = torch.cat([states, s.unsqueeze(0)], 0) 
    # offset += i+1
    output = torch.mean(states[-CONCAT:], dim=0) 
    return output







class RNNModelSearch(RNNModel):

    def __init__(self, *args):
        super(RNNModelSearch, self).__init__(*args)
        
        self._args = args
        
    def new(self):
        model_new = RNNModelSearch(*self._args)
        #for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        #    x.data.copy_(y.data)
        return model_new

         
    def _loss(self, hidden, input, target, criterion):
      log_prob, hidden_next = self(input, hidden, return_h=False) 
      
      loss = criterion(log_prob, target)
      return loss, hidden_next
  
