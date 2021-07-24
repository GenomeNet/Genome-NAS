#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 10:18:41 2021

@author: amadeu
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from generalNAS_tools.operations_14_9 import *
from torch.autograd import Variable
#from genotypes_cnn import PRIMITIVES_cnn, Genotype_cnn

#from genotypes_rnn import PRIMITIVES, STEPS, CONCAT, total_genotype
from generalNAS_tools.genotypes import PRIMITIVES_cnn, PRIMITIVES_rnn, rnn_steps, CONCAT, Genotype


#from genotypes_rnn import STEPS
from generalNAS_tools.utils import mask2d
from generalNAS_tools.utils import LockedDropout

from operator import itemgetter 


# import cnn_eval

INITRANGE = 0.04

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import math


class MixedOp(nn.Module):

    def __init__(self, C, stride, switch, p): # C, stride, switch=switches[switch_count], p=self.p
        super(MixedOp, self).__init__()
        self.m_ops = nn.ModuleList()
        self.p = p
        self.stride = stride
        self.switch = switch
        #for i in range(len(switch)): 
        #    if switch[i]: 
        #        primitive = PRIMITIVES_cnn[i]
        #        op = OPS[primitive](C, stride, False)
        #        if 'pool' in primitive:
        #            op = nn.Sequential(op, nn.BatchNorm1d(C, affine=False))
        #        if isinstance(op, Identity) and p > 0: #
        #            op = nn.Sequential(op, nn.Dropout(self.p))
        #        self.m_ops.append(op)
                #cells."numlayer".cell_ops."edges".m_ops."num_i's".op""
            #if switch[i] == False:
            #    op = None
            #    self.m_ops.append(op)
        for i in range(len(switch)): 
            # i=1
            if switch[i]: 
                primitive = PRIMITIVES_cnn[i]
                op = OPS[primitive](C, stride, False)
                if 'pool' in primitive:
                    op = nn.Sequential(op, nn.BatchNorm1d(C, affine=False))
                if isinstance(op, Identity) and p > 0: #
                    op = nn.Sequential(op, nn.Dropout(self.p))
                self.m_ops.add_module(primitive, op)
            else:
                op = None
                self.m_ops.add_module(op,op)
                           

                
    def update_p(self):
        for op in self.m_ops:
            if isinstance(op, nn.Sequential):
                if isinstance(op[0], Identity):
                    op[1].p = self.p
                    
    def forward(self, x, weights):
        if self.switch.count(False)==9: 
            # falls normal cell
            if self.stride == 1:
                # return x.mul(0.)
                return torch.zeros_like(x)
            return torch.zeros_like(x[:,:,::self.stride]) 
        else:
        
            return sum(w * op(x) for w, op in zip(weights, self.m_ops)) # vorher "self.m_ops" anstatt "m_ops_new"
       



class CNN_Cell_search(nn.Module):
    # steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, switches_normal, self.p
    # C_curr = C, switches_normal
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, switches, p):
        super(CNN_Cell_search, self).__init__()
        self.reduction = reduction
        self.p = p
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
            
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        
        self._steps = steps
        
        self._multiplier = multiplier

        self.cell_ops = nn.ModuleList()
                
        switch_count = 0
        discard_switch = []
        
        for i in range(self._steps):
         
            for j in range(2+i):
                # i0: j=0,1 ; count=0,1
                # i1: j=0,1,2 ; count=2,3,4
                # i2: j=0,1,2,3 ; count=5,6,7,8
                # i3: j=0,1,2,3,4 ; count = 9,10,11,12,13
                
                #if switches[switch_count].count(False) != len(switches[switch_count]): # if no discarded edge
                    #
                    stride = 2 if reduction and j < 2 else 1
                    op = MixedOp(C, stride, switch=switches[switch_count], p=self.p) # d.h. für i=1, greift er auf 0te, 1te und 2te zeile von switches
                    # opdict = op.state_dict() # 36 stück
                    # opdictnew = op.state_dict() # 18 stück
                    # self.cell_ops.append(op)
                    self.cell_ops.add_module(switch_count, op)
                    switch_count = switch_count + 1
                    
                #else: # of discarded edge
                #    discard_switch.append(switch_count)
                #    op = None
                #    self.cell_ops.append(op)
                    
                    
        #self.discard_switch = discard_switch
              
    
    def update_p(self):
        for op in self.cell_ops:
            op.p = self.p
            op.update_p()
            

    def forward(self, s0, s1, weights):
        
        #cell_ops_new = nn.ModuleList()
        #for op in self.cell_ops:
        #    if op != None:
        #        cell_ops_new.append(op)
        
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        #offset = 0
        #start = 0

        for i in range(self._steps):
            
            # i=3
            
            #node_states = []
            
            #for j, h in enumerate(states):
                
            #    if offset+j not in self.discard_switch:
            #        node_states.append(h)
            
            # end = len(node_states) + start
            
            #for j in range(start, end):
                
            #    s = sum(cell_ops[j](node_states[j-start], weights[j]))
                
            s = sum(self.cell_ops[self.w_pos[i]+j](h, weights[self.w_pos[i]+j]) for j, h in enumerate(itemgetter(*self.state_idxs[i])(states))) # vorher node_states und start anstatt offset
         
                
            #start = end
 
            # num_edges: 
            #offset += len(states) # 0, 2, 5, 9
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)
    

        #states = [1,1,2,1,2]        
        #weights=[0,1,2,3,4]
        
        #s = 1*0+2*1
        #s = sum(weights[0+j]*h for j, h in enumerate(itemgetter(*[0,2])(states)))
       
        

                
                  

    
    
   
# for evaluation
class DARTSCell(nn.Module):

  def __init__(self, ninp, nhid, dropouth, dropoutx, genotype):
    super(DARTSCell, self).__init__()
    self.nhid = nhid
    self.dropouth = dropouth
    self.dropoutx = dropoutx
    self.genotype = genotype
    #self.genotype = genotype[4]

    steps = len(self.genotype[4]) if self.genotype is not None else rnn_steps
    self._W0 = nn.Parameter(torch.Tensor(ninp+nhid, 2*nhid).uniform_(-INITRANGE, INITRANGE)) 
    self._Ws = nn.ParameterList([
        nn.Parameter(torch.Tensor(nhid, 2*nhid).uniform_(-INITRANGE, INITRANGE)) for i in range(steps) 
    ])
   

  def forward(self, inputs, hidden):
  
    # inputs = torch.rand(2,4,200)
    # hidden = hidden[0]
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
      xh_prev = torch.cat([x * x_mask, h_prev * h_mask], dim=-1) # entlang der channels zusammenfügen
    else:
      xh_prev = torch.cat([x, h_prev], dim=-1)
      
    c0, h0 = torch.split(xh_prev.mm(self._W0), self.nhid, dim=-1) 
    # c0, h0 = torch.split(xh_prev.mm(_W0), nhid, dim=-1) 

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
    states = [s0]
    for i, (name, pred) in enumerate(self.genotype[4]):
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
    output = torch.mean(torch.stack([states[i] for i in self.genotype[5]], -1), -1)
    return output



# sp=0
# C, num_classes, layers, criterion, switches_normal, switches_reduce, p, steps, multiplier, stem_multiplier = args.init_channels + int(add_width[sp]), num_classes, args.layers + int(add_layers[sp]), criterion, switches_normal_cnn, switches_reduce_cnn, float(drop_rate[sp]), 4,4,3
# _C, _num_classes, _layers, _criterion, switches_normal, switches_reduce, p, _steps, _multiplier, stem_multiplier = args.init_channels + int(add_width[sp]), num_classes, args.layers + int(add_layers[sp]), criterion, switches_normal_cnn, switches_reduce_cnn, float(drop_rate[sp]), 4,4,3
class RNNModel(nn.Module):

    def __init__(self, seq_len, 
                 dropouth=0.5, dropoutx=0.5, C=8, num_classes=4, layers=3, steps=4, multiplier=4, stem_multiplier=3, search=True, 
                 drop_path_prob=0.2, genotype=None, criterion=nn.CrossEntropyLoss().to(device), switches_normal=[], switches_reduce=[], switches_rnn=[], p=0.0,
                 alphas_normal=None, alphas_reduce=None, alphas_rnn=None):
       
        super(RNNModel, self).__init__()
        
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self.p = p
        self.dropouth = dropouth
        self.dropoutx = dropoutx
        
        if search == True:
        
            self.switches_normal = switches_normal
            self.switches_reduce = switches_reduce
            self.switches_rnn = switches_rnn
            self.num_ops_cnn = sum(switches_normal[0])
            self.num_ops_rnn = sum(switches_rnn[0])
            
            #self.alphas_normal = alphas_normal
            #self.alphas_reduce = alphas_reduce
            #self.alphas_rnn = alphas_rnn
            
            #self.weights = alphas_rnn
            
            #for rnn in self.rnns:
            #    rnn.weights = self.weights
          
            #self._arch_parameters = [
            #    self.alphas_normal,
            #    self.alphas_reduce,
            #    self.alphas_rnn,
            #    ]
    

        C_curr = stem_multiplier*C
        self.stem = nn.Sequential(
            nn.Conv1d(4, C_curr, 13, padding=6, bias=False),
            nn.BatchNorm1d(C_curr)
        )
    
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        
        for i in range(layers):
         
            #if i in [layers//3, 2*layers//3]:
            #    C_curr *= 2
            #    reduction = True
            #    
            #else:
            #    reduction = False
                
            # define cells for search or for evaluation of final architecture
            # if we are searching for best cell
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
                if search == True:
                    switches = self.switches_reduce
            else:
                reduction = False
                if search == True:
                    switches = self.switches_normal
                  
            if search == True:
                cell = CNN_Cell_search(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, switches, self.p) 
                  
            else:
                from darts_tools import cnn_eval

                cell = cnn_eval.CNN_Cell_eval(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev) 

              
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier*C_curr

        self.drop_path_prob = drop_path_prob
        
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
            from darts_tools import model_search
            cell_cls = model_search.DARTSCellSearch
            self.rnns = [cell_cls(ninp, nhid, dropouth, dropoutx, self.switches_rnn)]

       
        self.rnns = torch.nn.ModuleList(self.rnns)
        
        self.alphas_normal = alphas_normal
        self.alphas_reduce = alphas_reduce
        #self.alphas_rnn = alphas_rnn
            
        self.weights = alphas_rnn
        
        for rnn in self.rnns:
            rnn.weights = self.weights
          
        self._arch_parameters = [
           self.alphas_normal,
           self.alphas_reduce,
           self.weights,
           ]
        
        #print(num_neurons)
        #print(out_channels)

        self.decoder = nn.Linear(num_neurons*out_channels, 1000) # because we have flattened 
        
        self.classifier = nn.Linear(1000, num_classes)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.dropout_lin = nn.Dropout(p=0.1)

        self.init_weights()

        self.ninp = ninp
        self.nhid = nhid
        self.nhidlast = nhidlast
        #self.dropout = dropout
        #self.dropouti = dropouti
        #self.dropoute = dropoute
        #self.cell_cls = cell_cls
        #self._initialize_arch_parameters()

        
    def init_weights(self):
        
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-INITRANGE, INITRANGE)    

    def forward(self, input, hidden, return_h=False):
        
        s0 = s1 = self.stem(input)
        
        for i, cell in enumerate(self.cells):
           
            try: # when searching for cells
                if cell.reduction:
                    if self.alphas_reduce.size(1) == 1:
                        weights = F.softmax(self.alphas_reduce, dim=0)
                    else:
                        weights = F.softmax(self.alphas_reduce, dim=-1)
                else:
                    if self.alphas_normal.size(1) == 1:
                        weights = F.softmax(self.alphas_normal, dim=0)
                    else:
                        weights = F.softmax(self.alphas_normal, dim=-1)
                s0, s1 = s1, cell(s0, s1, weights)

                
            except: # when evaluating final architecture
                s0, s1 = s1, cell(s0, s1, self.drop_path_prob)  
                
        batch_size = s1.size(0) 

        out_channels = s1.size(1)
        num_neurons = s1.size(2)
        
        
        #CNN expects [batchsize, input_channels, signal_length]
        # RHN expect [seq_len, batch_size, features]
        # [2, 256, 50]
        x = s1.permute(2,0,1)
        # emb = torch.reshape(s1, (num_neurons,batch_size,out_channels)) 
        
        # x = torch.rand(10,2,256)
        # hidden = hidden[0] # [1,2,256] nur mit 0en: 1 weil für jeden timestep, 2 für batchsize und 256 für channels

        raw_output = x 
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
        
        output = output.permute(1,0,2)
                
        # flatten the RNN output
        x = torch.flatten(output, start_dim= 1) 
        
        x = self.dropout_lin(x)
    
        # linear layer
        x = self.decoder(x) 
        
        # dropout layer
        # x = self.dropout_lin(x)
        
        x = self.relu(x)
        
        # linear layer
        logit = self.classifier(x)

        if return_h:
            return logit, hidden, raw_outputs, outputs
        return logit, hidden 


    def update_p(self):
        for cell in self.cells:
            cell.p = self.p
            cell.update_p()
  
    
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return [Variable(weight.new(1, bsz, self.nhid).zero_())]