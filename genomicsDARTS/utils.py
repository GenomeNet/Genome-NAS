#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 10:26:52 2021

@author: amadeu
"""

import torch
import torch.nn as nn
import os, shutil
import numpy as np
from torch.autograd import Variable



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt

# targets = torch.tensor([2,1])
# output, target = log_prob, targets
def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    #print(k) 
    #correct_k = correct[:k].view(-1).float().sum(0)
    correct_k = correct[:k].reshape(k*batch_size).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res



def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.FloatTensor(x.size(0), 1, 1).bernoulli_(keep_prob)) # 80% of values are 1, and 20% are 0
    x.div_(keep_prob) # divides each element with 0.8
    x.mul_(mask) # multiplies ist with mask -> 20% are zeroed out
  return x


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)




def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def save_checkpoint(model, optimizer, epoch, path, finetune=False):
    if finetune:
        torch.save(model, os.path.join(path, 'finetune_model.pt'))
        torch.save(optimizer.state_dict(), os.path.join(path, 'finetune_optimizer.pt'))
    else:
        torch.save(model, os.path.join(path, 'model.pt'))
        torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer.pt'))
    torch.save({'epoch': epoch+1}, os.path.join(path, 'misc.pt'))

        


class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout, self).__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x


def mask2d(B, D, keep_prob, cuda=True): 

    
    m = torch.floor(torch.rand(B, D) + keep_prob) / keep_prob
    
    m = Variable(m, requires_grad=False)
    if cuda:
        #m = m.cuda()
        m = m.to(device)
    return m