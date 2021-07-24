#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 10:15:51 2021

@author: amadeu
"""

from generalNAS_tools.operations_14_9 import *
from generalNAS_tools.genotypes import PRIMITIVES_cnn, PRIMITIVES_rnn, rnn_steps, CONCAT
# from training_utils import mask2d
from generalNAS_tools.utils import mask2d

import generalNAS_tools.genotypes
import numpy as np
from torch.autograd import Variable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

INITRANGE = 0.04

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import math



class MixedOp(nn.Module):
    '''mask: op'''
    
    # mask = supernet_mask[1]
    def __init__(self, C, stride, mask_cnn):
        super(MixedOp, self).__init__()
        self.stride = stride
        # self._ops = nn.ModuleList()
        self.m_ops = nn.ModuleList()
        # mask_cnn = mask_cnn[0]
        mask_1 = np.nonzero(mask_cnn)[0] # 
        # mask_cnn = supernet_mask[0][0]
      
        self._super_mask = mask_1 # [3]
        #if len(mask_1) != 0: # wenn min 1ne op noch
        #    for selected in np.nditer(mask_1): # iteriere über alle ops
        #   
        #        primitive = PRIMITIVES_cnn[selected] # 3tes Element von Primitives_cnn
        #        op = OPS[primitive](C, stride, False)
        #        if 'pool' in primitive:
        #            op = nn.Sequential(op, nn.BatchNorm1d(C, affine=False))
        #        self._ops.append(op)
        
        for i in range(len(mask_cnn)): 
            if i in mask_1:
                primitive = PRIMITIVES_cnn[i]
                op = OPS[primitive](C, stride, False)
                if 'pool' in primitive:
                    op = nn.Sequential(op, nn.BatchNorm1d(C, affine=False))
                #if isinstance(op, Identity) and p > 0: #
                #    op = nn.Sequential(op, nn.Dropout(self.p))
                self.m_ops.append(op)
            if i not in mask_1:
                op = None
                self.m_ops.append(op)

    
    def forward(self, x, mask_cnn):
        #if (mask_cnn==0).all(): # falls diese Zeile (von insgesamt 14) von mask nur 0en hat, dann haben wir 0er Matrix
            # falls normal cell
        #    if self.stride == 1:
                # return x.mul(0.)
        #        return torch.zeros_like(x)
        #    return torch.zeros_like(x[:,:,::self.stride])
        #else: 
        #    result = 0
        #    mask_2 = np.nonzero(mask_cnn)[0] 
        #    if len(mask_2) != 0:
        #        for selected in np.nditer(mask_2): 
                    
        #             pos = np.where(mask_2==selected)[0][0] # pos ist 0, weil _super_mask oben nur 1 element hat 
        #              result += self._ops[pos](x)
                    
        #    return result
        
        m_ops_new = nn.ModuleList()
        for op in self.m_ops:
            if op != None:
                m_ops_new.append(op)
                
        return sum(op(x) for op in m_ops_new) #

        # return sum(w * op(x) for w, op in zip(weights, m_ops_new)) # vorher "self.m_ops" anstatt "m_ops_new"
        
        


class CNN_Cell_search(nn.Module):
    '''mask: 14 * 8'''

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, mask_cnn):
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
        cnt = 0
        
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, mask_cnn[cnt])
                
                self._ops.append(op) 
                
                cnt += 1

    def forward(self, s0, s1, mask_cnn):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        
        for i in range(self._steps):
            
            s = sum(self._ops[offset + j].forward(h, mask_cnn[offset + j]) for j, h in enumerate(states))
        
                
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)
    
    



# for evaluation
class DARTSCell(nn.Module):

  def __init__(self, ninp, nhid, dropouth, dropoutx, genotype):
    super(DARTSCell, self).__init__()
    self.nhid = nhid
    self.dropouth = dropouth
    self.dropoutx = dropoutx
    self.genotype = genotype

    steps = len(self.genotype.rnn) if self.genotype is not None else rnn_steps
    self._W0 = nn.Parameter(torch.Tensor(ninp+nhid, 2*nhid).uniform_(-INITRANGE, INITRANGE)) # [512,512]
    self._Ws = nn.ParameterList([
        nn.Parameter(torch.Tensor(nhid, 2*nhid).uniform_(-INITRANGE, INITRANGE)) for i in range(steps) # []
    ]) # liste mit 8 elementen und jedes Element ist ein weight mit shape [256, 512]
   
  # inputs = x
  # hidden = torch.rand(1,2,256)
  def forward(self, inputs, hidden):
  
    T, B = inputs.size(0), inputs.size(1) # 10,2

    if self.training:
        
      x_mask = mask2d(B, inputs.size(2), keep_prob=1.-self.dropoutx) 

      h_mask = mask2d(B, hidden.size(2), keep_prob=1.-self.dropouth)

    else:
        
      x_mask = h_mask = None
      
    hidden = hidden[0].to(device)

    hiddens = []
    
    for t in range(T):
 
      hidden = self.cell(inputs[t], hidden, x_mask, h_mask)
      hidden = hidden.to(device)
      hiddens.append(hidden)
      
    hiddens = torch.stack(hiddens)
    return hiddens, hiddens[-1].unsqueeze(0)

  def _compute_init_state(self, x, h_prev, x_mask, h_mask):

    if self.training:
      xh_prev = torch.cat([x * x_mask, h_prev * h_mask], dim=-1) # entlang der channels zusammenfügen
    else:
      xh_prev = torch.cat([x, h_prev], dim=-1) # x hat shape [2,512] und h_prev hat shape [2,512]
      
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
    for i, (name, pred) in enumerate(self.genotype.rnn):
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




# C, num_classes, layers, criterion, steps, multiplier, stem_multiplier = 16, 4, 3, nn.CrossEntropyLoss().to(device), 4, 4, 3 
# _C, _num_classes, _layers, _criterion, _steps, _multiplier, _stem_multiplier = 16, 4, 3, nn.CrossEntropyLoss().to(device), 4, 4, 3 

# x = torch.rand(2,4,200)
# x = torch.rand(10,2,256) # nach CNN_cells
# hidden = torch.rand(1,2,256)
# hidden = hidden[0] # [1,2,256] 
# ninp, nhid, nhidlast = 256, 256, 256

#args.seq_len, args.dropout, args.dropouth, args.dropoutx, args.dropouti, args.dropoute,
#                            args.init_channels, args.num_classes, args.layers, args.steps, args.multiplier,
#                            args.stem_multiplier, True, 0.2, None, mask

class RNNModel(nn.Module):

    def __init__(self, seq_len, 
                 dropouth=0.5, dropoutx=0.5,
                 C=8, num_classes=4, layers=3, steps=4, multiplier=4, stem_multiplier=3,
                 search=True, drop_path_prob=0.2, genotype=None, mask=None):
        super(RNNModel, self).__init__()
        
        
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        #self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        #self.p = p
        self.mask_cnn = mask[0]
        self.mask_rnn = mask[1]

    

        C_curr = stem_multiplier*C
        self.stem = nn.Sequential(
            nn.Conv1d(4, C_curr, 13, padding=6, bias=False),
            nn.BatchNorm1d(C_curr)
        )
    
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        
        for i in range(layers):
         
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
                
              # define cells for search or for evaluation of final architecture
            if search == True: 
                # if we are searching for best cell
                cell = CNN_Cell_search(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, self.mask_cnn) 
            else:
                # if we want to evaluate the best found architecture 
                cell = cnn_eval.CNN_Cell_eval(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, self.mask_cnn) 
              
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier*C_curr

        self.drop_path_prob = drop_path_prob
        
        
        out_channels = C_curr*steps
        
        num_neurons = math.ceil(math.ceil(seq_len / len([layers//3, 2*layers//3])) / 2)
        
        ninp, nhid, nhidlast = out_channels, out_channels, out_channels
    
    
        # x = torch.rand(10,2,256)
        # hidden = hidden[0] # [1,2,256]
            
        assert ninp == nhid == nhidlast
        # again, we have different rnn cells for search and for evaluation
        if search == False: 
            # rnn cell for evaluation of final architecture
            assert genotype is not None
            cell_cls = DARTSCell
            self.rnns = [cell_cls(ninp, nhid, dropouth, dropoutx, genotype)] # 
            # rnns = [cell_cls(ninp, nhid, dropouth, dropoutx, genotype)]

        else:
            # run search
            assert genotype is None
            from randomSearch_and_Hyperband_Tools import model_search
            cell_cls = model_search.DARTSCellSearch
            self.rnns = [cell_cls(ninp, nhid, dropouth, dropoutx, self.mask_rnn)]

       
        self.rnns = torch.nn.ModuleList(self.rnns)

        self.decoder = nn.Linear(num_neurons*out_channels, 700) # because we have flattened 
        
        self.classifier = nn.Linear(700, num_classes)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.dropout_lin = nn.Dropout(p=0.1)

        self.init_weights()

        self.ninp = ninp
        self.nhid = nhid
        self.nhidlast = nhidlast
      
        self.cell_cls = cell_cls

        
    def init_weights(self):
        
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-INITRANGE, INITRANGE)  
        
    def forward(self, input, hidden, return_h=False):
        
        s0 = s1 = self.stem(input)
        
        for i, cell in enumerate(self.cells):
            
            s0, s1 = s1, cell.forward(s0, s1, self.mask_cnn)
        
        batch_size = s1.size(0) 

        out_channels = s1.size(1)
        num_neurons = s1.size(2)
        
        #CNN expects [batchsize, input_channels, signal_length]
        # RHN expect [seq_len, batch_size, features]
        
        x = s1.permute(2,0,1)
        # emb = torch.reshape(s1, (num_neurons,batch_size,out_channels)) 

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
    
    def _loss(self, hidden, input, target):
        log_prob, hidden_next = self(input, hidden, return_h=False) 
          
        criterion = nn.CrossEntropyLoss()
        loss = criterion(log_prob, target)
        
        return loss, hidden_next
  
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return [Variable(weight.new(1, bsz, self.nhid).zero_())]



# init_channels, CIFAR_CLASSES, layers, mask = 8,10,5,supernet_mask
# _C, num_classes, _layers, mask, _steps, _multiplier, stem_multiplier = 8,10,5,supernet_mask,4,4,3
# C, num_classes, layers, mask, steps, multiplier, stem_multiplier = 8,10,5,supernet_mask,4,4,3



if __name__ == "__main__":
    num_ops = 8
    steps = 4
    arch = "BONAS"
    genotype = eval("genotypes.%s" % arch)
    mask = geno2mask(genotype)

    print(str(mask))
    print(hash(mask.tostring()))
    k = sum(1 for i in range(steps) for n in range(2 + i))
    mask = torch.ones(k, num_ops)

    '''---------------------------------------------'''
    mask = np.array([[0, 1, 0, 0, 1, 1, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 1, 1, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 1, 0, 0],
                     [0, 0, 0, 0, 0, 1, 1, 1]]
                    )
    # print(mask)
    mix_op = MixedOp(C=16, stride=1, mask=mask[0])
    cell = Cell(steps, multiplier=4, C_prev_prev=16, C_prev=16, C=16, reduction=False, reduction_prev=False, mask=mask)
