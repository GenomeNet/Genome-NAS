#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 10:16:14 2021

@author: amadeu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
#from genotypes_rnn import PRIMITIVES, STEPS, CONCAT, total_genotype
from torch.autograd import Variable
from collections import namedtuple
# from model import DARTSCell, RNNModel
from darts_tools.model import DARTSCell, RNNModel

import numpy as np

from generalNAS_tools.operations_14_9 import *
#from genotypes_cnn import PRIMITIVES_cnn
#from genotypes_cnn import Genotype_cnn
from generalNAS_tools.genotypes import PRIMITIVES_cnn, PRIMITIVES_rnn, rnn_steps, CONCAT, Genotype

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from darts_tools.genotype_parser import parse_genotype

#import cnn_eval

# for search
class DARTSCellSearch(DARTSCell):

  def __init__(self, ninp, nhid, dropouth, dropoutx, switch_rnn):
    super(DARTSCellSearch, self).__init__(ninp, nhid, dropouth, dropoutx, genotype=None)
    self.bn = nn.BatchNorm1d(nhid, affine=False)
    self.switch_rnn = switch_rnn
    
    # probs = model.arch_parameters()[2]
    # hidden = torch.rand(2,256)
    # x = torch.rand(4,200)
    # h_mask = torch.rand(4,256)
    # x_mask = torch.rand(4,200)

    
  def cell(self, x, h_prev, x_mask, h_mask):
      
    # print(x.shape)
    # print(x_mask.shape)
    # print(h_prev.shape)
    # print(h_mask.shape)
    # x_mask = h_mask = x = h_prev = torch.rand(2,256)
    # probs = alphas_rnn
    # s0 = torch.rand(2,256)

      
    s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)
    s0 = self.bn(s0)
    probs = F.softmax(self.weights, dim=-1) 
   
    offset = 0
    states = s0.unsqueeze(0) 
    disc_edge_count = 0
    
    for i in range(rnn_steps):
      # 
      if self.training: 
        masked_states = states * h_mask.unsqueeze(0) 
      else:
        masked_states = states
        
      ch = masked_states.view(-1, self.nhid).mm(self._Ws[i]).view(i+1, -1, 2*self.nhid) 
      c, h = torch.split(ch, self.nhid, dim=-1)
      
      # ch = masked_states.view(-1, nhid).mm(_Ws[i]).view(i+1, -1, 2*nhid) 
      # c, h = torch.split(ch, nhid, dim=-1)
       
      c = c.sigmoid()
    
      s = torch.zeros_like(s0) 
      
      for j in range(offset, offset+i+1):
        # j=0
        count_false = 0
        rnn_switch_row = self.switch_rnn[j]
        
        if rnn_switch_row.count(False) != len(rnn_switch_row):
            
            for k, name in enumerate(PRIMITIVES_rnn): # geht von 0:4
              # switch_rnn = switches_rnn[0]
              # k=4
              #print('ctn')
              #print(count_false)
              if rnn_switch_row[k]:
                
                name = PRIMITIVES_rnn[k]
                
                if name == 'none':
                  continue
                fn = self._get_activation(name)
                # fn = _get_activation(name)
    
                unweighted = states + c * (fn(h) - states)
                  
                s += torch.sum(probs[j-disc_edge_count, k-count_false].unsqueeze(-1).unsqueeze(-1) * unweighted, dim=0)  
              else:
                count_false += 1
        else:          
            disc_edge_count += 1
          
      s = self.bn(s) 
      states = torch.cat([states, s.unsqueeze(0)], 0) 
      offset += i+1
    output = torch.mean(states[-CONCAT:], dim=0) 
    return output

 

alphas_final = []

start = time.time()
for i in range(rnn_steps):
    
    alphas_step = alphas_rnn[offset:offset+i+1]

    for k, name in enumerate(PRIMITIVES_rnn):
        
        alphas_aux = []

        for j in range(offset, offset+i+1):
     
            if switch_rnn[j][k]:
                
                alphas_aux.append(alphas_rnn[j,k])#torch.cat(alphas_aux, alphas_step[j,k])
                
        alphas_final.append(alphas_aux)
        


        
training=True
offset = 0
states = s0.unsqueeze(0) 
disc_edge_count = 0
start_time = time.time()

for i in range(rnn_steps):
    # i=1
    if training: 
        masked_states = states * h_mask.unsqueeze(0) 
    else:
        masked_states = states
        
    #ch = masked_states.view(-1, self.nhid).mm(self._Ws[i]).view(i+1, -1, 2*self.nhid) 
    #c, h = torch.split(ch, self.nhid, dim=-1)
      
    ch = masked_states.view(-1, nhid).mm(_Ws[i]).view(i+1, -1, 2*nhid) 
    c, h = torch.split(ch, nhid, dim=-1)
       
    c = c.sigmoid()
    
    s = torch.zeros_like(s0) 
    s = s.unsqueeze(0)
    unweighteds = [states + c * (torch.tanh(h) - states), states + c * (torch.sigmoid(h) - states), states + c * (torch.relu(h) - states), states + c * (h - states)]

      
    for j in range(offset, offset+i+1):
       # j=1
       #count_false = 0
       #rnn_switch_row = self.switch_rnn[j]
               
       s += sum(w.unsqueeze(-1).unsqueeze(-1) * op for w, op in zip(alphas_rnn[j,1:5], unweighteds)) # s [2,256] immer ; unw [1,2,256] bei i=0, unw [2,2,256]

    states = torch.cat([states, s], 0) # von [1,2,256] zu [2,2,256]
    offset += i+1
     
end_time = time.time()
duration = end_time - start_time


s += torch.sum(probs[j-disc_edge_count, k-count_false].unsqueeze(-1).unsqueeze(-1) * unweighted, dim=0)  

r=s
f=s

r += torch.sum(alphas_rnn[0:2,0].unsqueeze(-1).unsqueeze(-1) * unw, dim=0)
  
f += torch.sum(probs[rows[i], cols[i]].unsqueeze(-1).unsqueeze(-1) * unw, dim=0)

i=0
rows = [[0,1,2],[3,4,5,6]]
cols = [[0],[0]]


                    
        

    


                

            
                    
                

    


class RNNModelSearch(RNNModel):

    def __init__(self, *args):
        super(RNNModelSearch, self).__init__(*args)
        
        self._args = args
        # self._initialize_arch_parameters()
        
    def new(self):
        model_new = RNNModelSearch(*self._args)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

         
    def _loss(self, hidden, input, target):
      log_prob, hidden_next = self(input, hidden, return_h=False) 
      criterion = nn.CrossEntropyLoss()
      loss = criterion(log_prob, target)
      return loss, hidden_next
  
    def _initialize_arch_parameters(self):

      # alphas for cnn
      k_cnn = sum(1 for i in range(self._steps) for n in range(2+i))
       
      num_ops_cnn = self.num_ops_cnn
        
      self.alphas_normal = nn.Parameter(torch.FloatTensor(1e-3*np.random.randn(k_cnn, num_ops_cnn)))
      self.alphas_reduce = nn.Parameter(torch.FloatTensor(1e-3*np.random.randn(k_cnn, num_ops_cnn)))  
        
      # alphas for rnn
      k_rnn = sum(i for i in range(1, rnn_steps+1)) # 1+2+3+4+5+6+7+8=36
        
      num_ops_rnn = self.num_ops_rnn
    
         
      self.weights = nn.Parameter(torch.FloatTensor(1e-3*np.random.randn(k_rnn, num_ops_rnn)))
    
      self.alphas_rnn = self.weights
      for rnn in self.rnns:
          rnn.weights = self.weights
          
      self._arch_parameters = [
          self.alphas_normal,
          self.alphas_reduce,
          self.alphas_rnn,
          ]
          
          
    def arch_parameters(self):
        return self._arch_parameters

       
            

    def genotype(self):
        
        genotype = parse_genotype(self.switches_normal, self.alphas_normal, self.switches_reduce, self.alphas_reduce, self.switches_rnn, self.weights)

        return genotype