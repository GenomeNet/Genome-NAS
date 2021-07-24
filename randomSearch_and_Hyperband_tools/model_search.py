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

import numpy as np

#from operations import *
#from genotypes_cnn import PRIMITIVES_cnn
#from genotypes_cnn import Genotype_cnn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#import cnn_eval

# for search

# ninp, nhid, dropouth, dropoutx, mask
class DARTSCellSearch(DARTSCell):

  def __init__(self, ninp, nhid, dropouth, dropoutx, mask_rnn):
    super(DARTSCellSearch, self).__init__(ninp, nhid, dropouth, dropoutx, genotype=None)
    
    self.bn = nn.BatchNorm1d(nhid, affine=False)
    self.mask_rnn = mask_rnn

  # hidden = cell(inputs[t], hidden, x_mask, h_mask) # hidden, x_mask und h_mask hat jetzt auch shape [2,256]
  # t=0
  # x = inputs[t] # shape [2,256] 
  # hidden, x = torch.rand(2,256), torch.rand(2,256)
  def cell(self, x, h_prev, x_mask, h_mask):
      
    
    s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)
    #print(s0.shape)
    s0 = self.bn(s0)
    #probs = F.softmax(self.weights, dim=-1) 
   
    offset = 0
    states = s0.unsqueeze(0) # [1,2,256]
    
    for i in range(rnn_steps):
      # i = 3
    
      if self.training:
        masked_states = states * h_mask.unsqueeze(0) 
      else:
        masked_states = states # bei i=0 [1,2,256], bei i=1 [2,2,256]
      ch = masked_states.view(-1, self.nhid).mm(self._Ws[i]).view(i+1, -1, 2*self.nhid) # [1,2,256] wird mit dem i-ten element von _Ws multipliziert also die 8 elmentige weight  [256, 512] geht er durch 
      # ch = masked_states.view(-1, nhid).mm(_Ws[i]).view(i+1, -1, 2*nhid) 

      c, h = torch.split(ch, self.nhid, dim=-1)
      
      # ch = masked_states.view(-1, nhid).mm(_Ws[i]).view(i+1, -1, 2*nhid) 
      # c, h = torch.split(ch, nhid, dim=-1) # beides [1,2,256] shape bei i=0, [2,2,256] bei i=1, [3,2,256] bei i=2, [8,2,256] bei i=7
       
      c = c.sigmoid()
    
      s = torch.zeros_like(s0) # [2,256] nur mit 0en bei jedem i !
      
      for j in range(offset, offset+i+1):
       # j=7
       mask_row = self.mask_rnn[j] # wähle Zeile von mask
       # mask_row = mask_rnn[j] # wähle Zeile von mask

       mask_idx = np.nonzero(mask_row)[0]
       
       if (mask_row==0).all(): # dann wäre s immer noch nur [2,256] mit nur 0en
           continue
          
       # for k, name in enumerate (PRIMITIVES):  
       for op_idx in np.nditer(mask_idx):
         # op_idx = 2
         name = PRIMITIVES_rnn[op_idx]
         if name == 'none':
           continue
         fn = self._get_activation(name)
         # fn = _get_activation(name)
             
         unweighted = states + c * (fn(h) - states) 
         # states, unweighted, c und h ändern sich nach jeder der 8er for-loop, bei i=0 [1,2,256] shape, bei i=1 [2,2,256] usw... bei i=7 [8,2,256]
         # i=0: unweighted= [1,2,256]: also output wirklich immer [1,2,256] egal wie viele operationen
         # i=1: unweighted hat immer shape [2,2,256]
         # i=2: unweighted hat immer shape [3,2,256]
         
         s += torch.sum(unweighted, dim=0)  # bleibt immer [2, 256] egal wie viele ops und welches i

         #s += torch.sum(probs[j, op_idx].unsqueeze(-1).unsqueeze(-1) * unweighted, dim=0)  
             
      s = self.bn(s) 
      states = torch.cat([states, s.unsqueeze(0)], 0) # vor i=0 ist states noch [1,2,256] aber wird nach i=0 zu [2,2,256], nach i=1 zu [3,2,256] usw. bis [8,2,256] erreicht wird
      # und dies alles unabhängig davon, wie viele operationen
      offset += i+1
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

         
    def _loss(self, hidden, input, target):
      log_prob, hidden_next = self(input, hidden, return_h=False) 
      
      criterion = nn.CrossEntropyLoss()
      loss = criterion(log_prob, target)
      return loss, hidden_next