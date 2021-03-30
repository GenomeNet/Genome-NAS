#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 10:26:16 2021

@author: amadeu
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from genotypes_rnn import STEPS
from utils import mask2d
from utils import LockedDropout
#from utils import embedded_dropout
from torch.autograd import Variable

import cnn_eval

import math

INITRANGE = 0.04

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from operations import *
from torch.autograd import Variable
from genotypes_cnn import PRIMITIVES_cnn
from genotypes_cnn import Genotype_cnn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES_cnn: 
   
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive: 
        op = nn.Sequential(op, nn.BatchNorm1d(C, affine=False))

      self._ops.append(op) 

  def forward(self, x, weights):
  
    return sum(w * op(x) for w, op in zip(weights, self._ops)) 
  


class CNN_Cell_search(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(CNN_Cell_search, self).__init__()
    self.reduction = reduction

    if reduction_prev: 
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False) 
      
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps): 
   
      for j in range(2+i):
      
        stride = 2 if reduction and j < 2 else 1
        
        op = MixedOp(C, stride) 
        self._ops.append(op) 

  def forward(self, s0, s1, weights): 
    s0 = self.preprocess0(s0) 
    s1 = self.preprocess1(s1)
  
    states = [s0, s1]
    offset = 0
    
    for i in range(self._steps):
        
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states)) 
    
      offset += len(states)
      states.append(s) 

    return torch.cat(states[-self._multiplier:], dim=1) 


class DARTSCell(nn.Module):

  def __init__(self, ninp, nhid, dropouth, dropoutx, genotype):
    super(DARTSCell, self).__init__()
    self.nhid = nhid
    self.dropouth = dropouth
    self.dropoutx = dropoutx
    self.genotype = genotype

    steps = len(self.genotype.recurrent) if self.genotype is not None else STEPS
    self._W0 = nn.Parameter(torch.Tensor(ninp+nhid, 2*nhid).uniform_(-INITRANGE, INITRANGE)) 
    self._Ws = nn.ParameterList([
        nn.Parameter(torch.Tensor(nhid, 2*nhid).uniform_(-INITRANGE, INITRANGE)) for i in range(steps) 
    ])
   

  def forward(self, inputs, hidden):
  

    T, B = inputs.size(0), inputs.size(1) # 10,2

    if self.training:
        
      x_mask = mask2d(B, inputs.size(2), keep_prob=1.-self.dropoutx) 

      h_mask = mask2d(B, hidden.size(2), keep_prob=1.-self.dropouth)
    

    else:
        
      x_mask = h_mask = None
      
    hidden = hidden[0]


    hiddens = []
    for t in range(T):
 
      hidden = self.cell(inputs[t], hidden, x_mask, h_mask)

      hiddens.append(hidden)
    hiddens = torch.stack(hiddens)
    return hiddens, hiddens[-1].unsqueeze(0)

  def _compute_init_state(self, x, h_prev, x_mask, h_mask):

    if self.training:
      xh_prev = torch.cat([x * x_mask, h_prev * h_mask], dim=-1) 
    else:
    
      xh_prev = torch.cat([x, h_prev], dim=-1)
    c0, h0 = torch.split(xh_prev.mm(self._W0), self.nhid, dim=-1) 
  

    c0 = c0.sigmoid() 
    h0 = h0.tanh() 
    s0 = h_prev + c0 * (h0-h_prev)
    return s0

  def _get_activation(self, name):
    if name == 'tanh':
      f = torch.tanh
    elif name == 'relu':
      f = torch.relu
    elif name == 'sigmoid':
      f = torch.sigmoid
    elif name == 'identity':
      f = lambda x: x
    else:
      raise NotImplementedError
    return f

  def cell(self, x, h_prev, x_mask, h_mask):
    s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)
    print(s0)
    states = [s0]
    for i, (name, pred) in enumerate(self.genotype.recurrent):
      s_prev = states[pred]
      if self.training:
        ch = (s_prev * h_mask).mm(self._Ws[i])
      else:
        ch = s_prev.mm(self._Ws[i])
      c, h = torch.split(ch, self.nhid, dim=-1)
      c = c.sigmoid()
      fn = self._get_activation(name)
      h = fn(h)
      s = s_prev + c * (h-s_prev)
      states += [s]
    output = torch.mean(torch.stack([states[i] for i in self.genotype.concat], -1), -1)
    return output


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, seq_len, #ninp, nhid, nhidlast, 
                 dropout=0.5, dropouth=0.5, dropoutx=0.5, dropouti=0.5, dropoute=0.1,
                  C=8, num_classes=4, layers=3, steps=4, multiplier=4, stem_multiplier=3,
                 search=True, drop_path_prob=0.2, genotype=None):
        super(RNNModel, self).__init__()
        
        # dropout, dropouth, dropoutx, dropouti, dropoute, _C, _num_classes, _layers, _steps, _multiplier, stem_multiplier = args.dropout, args.dropouth, args.dropoutx, args.dropouti, args.dropoute,args.init_channels, args.num_classes, args.layers, args.steps, args.multiplier, args.stem_multiplier
        # dropout, dropouth, dropoutx, dropouti, dropoute, C, num_classes, layers, steps, multiplier, stem_multiplier = args.dropout, args.dropouth, args.dropoutx, args.dropouti, args.dropoute,args.init_channels, args.num_classes, args.layers, args.steps, args.multiplier, args.stem_multiplier
        self._C = C 
        self._num_classes = num_classes 
        self._layers = layers 
        self._steps = steps
        self._multiplier = multiplier
        
        C_curr = stem_multiplier*C 
        self.stem = nn.Sequential( 
          nn.Conv1d(4, C_curr, 3, padding=1, bias=False), 
          nn.BatchNorm1d(C_curr)
        )
         
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C 
        
     
        self.cells = nn.ModuleList()
        
        reduction_prev = False
        
        for i in range(layers): 
            
          #i = 2
          if i in [layers//3, 2*layers//3]: # if reduction layer
            C_curr *= 2
            reduction = True
          else:
            reduction = False
            
          # define cells for search or for evaluation of final architecture
          if search == True: 
              # if we are searching for best cell
              cell = CNN_Cell_search(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev) 
          else:
              # if we want to evaluate the best found architecture 
              cell = cnn_eval.CNN_Cell_eval(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev) 
              
          reduction_prev = reduction
          
          self.cells += [cell]
        
          C_prev_prev, C_prev = C_prev, multiplier*C_curr
         
        
        self.drop_path_prob=drop_path_prob
        
        
        out_channels = C_curr*steps
        
        num_neurons = math.ceil(math.ceil(seq_len / len([layers//3, 2*layers//3])) / 2)
        
        ninp, nhid, nhidlast = out_channels, out_channels, out_channels
    
            
        assert ninp == nhid == nhidlast
        # again, we have different rnn cells for search and for evaluation
        if search == False: 
            # rnn cell for evaluation of final architecture
            assert genotype is not None
            cell_cls = DARTSCell
            self.rnns = [cell_cls(ninp, nhid, dropouth, dropoutx, genotype)] # 
            #rnns = [cell_cls(ninp, nhid, dropouth, dropoutx, genotype)]

        else:
            # run search
            assert genotype is None
            import model_search
            cell_cls = model_search.DARTSCellSearch
            self.rnns = [cell_cls(ninp, nhid, dropouth, dropoutx)]

       
        self.rnns = torch.nn.ModuleList(self.rnns)

        self.decoder = nn.Linear(num_neurons*out_channels, 100) # because we have flattened 
        
        self.classifier = nn.Linear(100, num_classes)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.dropout_lin = nn.Dropout(p=0.1)

        self.init_weights()

        self.ninp = ninp
        self.nhid = nhid
        self.nhidlast = nhidlast
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropoute = dropoute
        self.cell_cls = cell_cls

    def init_weights(self):
        
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-INITRANGE, INITRANGE)
 

    def forward(self, input, hidden, return_h=False):
        # input, hidden = cur_data, hidden[s_id]
        
        input = input.transpose(1, 2).float()
                
        s0 = s1 = self.stem(input) 
    
        for i, cell in enumerate(self.cells):
          try: #
              if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)  
                #weights = F.softmax(model.alphas_normal, dim=-1)
              else:
                weights = F.softmax(self.alphas_normal, dim=-1)
                #weights = F.softmax(model.alphas_reduce, dim=-1)
              
              s0, s1 = s1, cell(s0, s1, weights)
          except: 
              s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
          
        batch_size = s1.size(0) 

        out_channels = s1.size(1)
        num_neurons = s1.size(2)
        
        emb = torch.reshape(s1, (num_neurons,batch_size,out_channels)) 

        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns): 
            
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            
        hidden = new_hidden

        output = raw_output
        
        outputs.append(output)
        
        # flatten the RNN output
        output = torch.reshape(output, (batch_size,num_neurons*out_channels)) 
      
        output = self.dropout_lin(output)
        
        x = self.decoder(output) 
        
        x = self.dropout_lin(x)
        
        x = self.relu(x)
        
        logit = self.classifier(x)

        if return_h:
            return logit, hidden, raw_outputs, outputs
        return logit, hidden 

    def init_hidden(self, bsz):
      weight = next(self.parameters()).data
      return [Variable(weight.new(1, bsz, self.nhid).zero_())]