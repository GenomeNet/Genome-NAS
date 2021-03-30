#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 10:25:39 2021

@author: amadeu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from genotypes_rnn import PRIMITIVES, STEPS, CONCAT, total_genotype
from torch.autograd import Variable
from collections import namedtuple
from model import DARTSCell, RNNModel

from operations import *
from genotypes_cnn import PRIMITIVES_cnn
from genotypes_cnn import Genotype_cnn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import cnn_eval


class DARTSCellSearch(DARTSCell):

  def __init__(self, ninp, nhid, dropouth, dropoutx):
    super(DARTSCellSearch, self).__init__(ninp, nhid, dropouth, dropoutx, genotype=None)
    self.bn = nn.BatchNorm1d(nhid, affine=False)

  def cell(self, x, h_prev, x_mask, h_mask):
    s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)
    s0 = self.bn(s0)
    probs = F.softmax(self.weights, dim=-1) 
   

    offset = 0
    states = s0.unsqueeze(0) 
    for i in range(STEPS):
    
      if self.training:
        masked_states = states * h_mask.unsqueeze(0) 
      else:
        masked_states = states
      ch = masked_states.view(-1, self.nhid).mm(self._Ws[i]).view(i+1, -1, 2*self.nhid) 
      c, h = torch.split(ch, self.nhid, dim=-1)
   
      c = c.sigmoid()

      s = torch.zeros_like(s0) 
      for k, name in enumerate(PRIMITIVES): 
        if name == 'none':
          continue
        fn = self._get_activation(name)
        unweighted = states + c * (fn(h) - states)
        
        s += torch.sum(probs[offset:offset+i+1, k].unsqueeze(-1).unsqueeze(-1) * unweighted, dim=0) 
        
      s = self.bn(s) 
      states = torch.cat([states, s.unsqueeze(0)], 0) 
      offset += i+1
    output = torch.mean(states[-CONCAT:], dim=0) 
    return output


class RNNModelSearch(RNNModel):

    def __init__(self, *args):
        super(RNNModelSearch, self).__init__(*args, genotype=None)
        self._args = args
        self._initialize_arch_parameters()


    def new(self):
        model_new = RNNModelSearch(*self._args)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def _initialize_arch_parameters(self):
      
      k_cnn = sum(1 for i in range(self._steps) for n in range(2+i))
   
      num_ops = len(PRIMITIVES_cnn) 
        
      self.alphas_normal = Variable(1e-3*torch.randn(k_cnn, num_ops).to(device), requires_grad=True) #
      self.alphas_reduce = Variable(1e-3*torch.randn(k_cnn, num_ops).to(device), requires_grad=True)
      
      k_rnn = sum(i for i in range(1, STEPS+1)) # 1+2+3+4+5+6+7+8=36
      
      weights_data = torch.randn(k_rnn, len(PRIMITIVES)).mul_(1e-3)
      
      self.weights = Variable(weights_data.to(device), requires_grad=True)

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

    def _loss(self, hidden, input, target):
      log_prob, hidden_next = self(input, hidden, return_h=False) 
      
      criterion = nn.CrossEntropyLoss()
      loss = criterion(log_prob, target)
      return loss, hidden_next


    def genotype(self):

      
      def _parse_cnn(weights):
            gene_cnn = []
            n = 2
            start = 0
            for i in range(self._steps):
                
                end = start + n 
                W = weights[start:end].copy() 
              
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES_cnn.index('none')))[:2]
                
                for j in edges: 
                    
                  k_best = None
                  for k in range(len(W[j])): 
              
                    if k != PRIMITIVES_cnn.index('none'): 
                      if k_best is None or W[j][k] > W[j][k_best]: 
                        k_best = k
                    
                  gene_cnn.append((PRIMITIVES_cnn[k_best], j))
                start = end
                n += 1
            return gene_cnn
            
      gene_normal = _parse_cnn(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
      #gene_normal = _parse(F.softmax(alphas_normal, dim=-1).data.cpu().numpy())
            
      gene_reduce = _parse_cnn(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())
      #gene_reduce = _parse(F.softmax(alphas_reduce, dim=-1).data.cpu().numpy())
            
      concat = range(2+self._steps-self._multiplier, self._steps+2)
      
      
      def _parse_rnn(probs):
        gene_rnn = []
        start = 0
        for i in range(STEPS):
          end = start + i + 1
          W = probs[start:end].copy()
          j = sorted(range(i + 1), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[0]
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene_rnn.append((PRIMITIVES[k_best], j))
          start = end
        return gene_rnn

      gene_rnn = _parse_rnn(F.softmax(self.weights, dim=-1).data.cpu().numpy())
    
     
      genotype = total_genotype(normal=gene_normal, normal_concat=concat, reduce=gene_reduce, reduce_concat=concat,
                                recurrent=gene_rnn, concat=range(STEPS+1)[-CONCAT:])
      
      return genotype