#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 21:45:38 2021

@author: amadeu
"""

import torch
import numpy as np
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1 Möglichkeit - ohne weight
true = torch.Tensor([[1,1,0,0,],
                     [1,0,1,0,],
                     [0,0,0,0,],
                     [0,0,0,1,],
                     [0,0,0,0,]])


pred = torch.Tensor([[2.1,-0.1,-0.1,-0.1],
                     [2.1,-0.1,-0.1,-0.1],
                     [-0.2,-0.1,-0.1,-0.1],
                     [-0.2,-0.1,-0.1,-0.1],
                     [-0.2,-0.1,-0.1,-0.1]])

final = nn.Sigmoid()

pred = final(pred)


preds = np.round(pred)
criterion = nn.BCELoss()
loss = criterion(true, pred)

# 2 Möglichkeit - mit weight für jeden sample
true = torch.Tensor([[1,1,0,0,],
                     [1,0,1,0,],
                     [0,0,0,0,],
                     [0,0,0,1,],
                     [0,0,0,0,]])


pred = torch.Tensor([[2.1,-0.1,-0.1,-0.1],
                     [2.1,-0.1,-0.1,-0.1],
                     [-0.2,-0.1,-0.1,-0.1],
                     [-0.2,-0.1,-0.1,-0.1],
                     [-0.2,-0.1,-0.1,-0.1]])

final = nn.Sigmoid()

pred = final(pred)

criterion = nn.BCEWithLogitsLoss(reduction='none')

# pred = torch.rand(64,919)
# true = torch.ones(64,919)

# num_classes = 919
# num_samples = 64
def sample_weighted_BCE(pred, true, criterion, num_classes, num_samples):
    # num_classes = true.size(1)
    # num_samples = true.size(0)
    class_counts = []
    weights = torch.zeros(num_samples, num_classes)
    for i in range(num_classes):
        # i = 0
        pos = torch.count_nonzero(true[:,i])
        neg = num_samples-pos
        for j in range(num_samples):
            # j=0
            if true[j,i] == 1.0:
                weights[j,i] = num_samples/pos
            else:
                weights[j,i] = num_samples/neg
                
    loss = criterion(pred, true)
    loss = (loss * weights.to(device)).mean()
    return loss 
    

# 3 Möglichkeit - wie bei Medium artikel, mit positive und negative weights
true = torch.Tensor([[1,1,0,0,],
                     [1,0,1,0,],
                     [0,0,0,0,],
                     [0,0,0,1,],
                     [0,0,0,0,]])


pred = torch.Tensor([[2.1,-0.1,-0.1,-0.1],
                     [2.2,2.1,-0.1,-0.1],
                     [2,-0.1,-0.1,-0.1],
                     [float("Inf"),-0.1,-0.1,-0.1],
                     [2,-0.1,-0.1,-0.1]])

NaN = float("NaN")
infinity = float("Inf")

final = nn.Sigmoid()

pred = final(pred)


   

class weighted_BCE(nn.Module):
    def __init__(self, train_queue, num_steps):
        super(weighted_BCE, self).__init__()
        
        trues = []
        for step, (input, target) in enumerate(train_queue): 
        
            if step > num_steps:
                break        
        
            trues.append(target)
        
        trues = torch.cat(trues)
        
        self.num_classes = target.size(1)
        self.num_samples = trues.size(0)
       
        positive_weights = []
        negative_weights = []
        for c in range(self.num_classes):
            # c=0
            pos = torch.count_nonzero(trues[:,c])
            neg = self.num_samples-pos
            positive_weights.append(self.num_samples/(pos))
            negative_weights.append(self.num_samples/(neg))
        
        positive_weights = torch.Tensor(positive_weights)
        negative_weights = torch.Tensor(negative_weights)
        max_weight = max(positive_weights[positive_weights != np.inf])
        positive_weights[np.where(positive_weights==np.inf)] = max_weight
        
        self.positive_weights = positive_weights.to(device)
        self.negative_weights = negative_weights.to(device)

    def forward(self, pred, true):
        
        pred = torch.clamp(pred, min=1e-8, max=1-1e-8)
    
       
        loss = self.positive_weights * (true * torch.log(pred)) + \
               self.negative_weights * ((1 - true) * torch.log(1 - pred))
    
        
        return torch.neg(torch.mean(loss))




# 4 Möglichkeit - mit pos_weight und BCEWithLogits

true = torch.Tensor([[1,1,0,0,],
                     [1,0,1,0,],
                     [0,0,0,0,],
                     [0,0,0,1,],
                     [0,0,0,0,]])


pred = torch.Tensor([[2.1,2.1,-0.1,-0.1],
                     [-0.2,-0.1,-0.1,-0.1],
                     [-0.2,-0.1,-0.1,-0.1],
                     [-0.2,-0.1,-0.1,-0.1],
                     [-0.2,-0.1,-0.1,-0.1]])


def get_criterion(train_queue, num_steps):
        
    trues = []
    for step, (input, target) in enumerate(train_queue): 
    
        if step > num_steps:
            break        
    
        trues.append(target)
    
    trues = torch.cat(trues)
    
    num_classes = target.size(1)
    num_samples = trues.size(0)
    
    class_counts = []
    
    for i in range(num_classes):
        class_counts.append(torch.count_nonzero(trues[:,i]))
    class_counts = np.array(class_counts)
    
    # class_counts = np.array([2,1,1,1])
    pos_weights = np.ones_like(class_counts)
    neg_counts = [num_samples-pos_count for pos_count in class_counts]
    
    for cdx, (pos_count, neg_count) in enumerate(zip(class_counts, neg_counts)):
            pos_weights[cdx] = neg_count / (pos_count + 1e-5)
           
    pos_weights = torch.Tensor(pos_weights).to(device)
    
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights).to(device)
    
    return criterion
   
        



loss = criterion(true, pred)



def get_pos_weight(num_classes, true):
    
    class_counts = []
    for i in range(num_classes):
        class_counts.append(torch.count_nonzero(true[:,i]))
    class_counts = np.array(class_counts)
    
    # class_counts = np.array([2,1,1,1])
    pos_weights = np.ones_like(class_counts)
    neg_counts = [len(true)-pos_count for pos_count in class_counts]
    for cdx, (pos_count, neg_count) in enumerate(zip(class_counts,  neg_counts)):
            pos_weights[cdx] = neg_count / (pos_count + 1e-5)
           
    pos_weights = torch.Tensor(pos_weights)


    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    
    return criterion



