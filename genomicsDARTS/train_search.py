import argparse
import os, sys, glob
import time
import math
import numpy as np
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from architect import Architect

import genotypes_rnn


import gc

import data
import model_search as model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from utils import repackage_hidden, create_exp_dir, save_checkpoint # batchify, get_batch,
import utils

import data_preprocessing as dp


parser = argparse.ArgumentParser(description='DARTS for genomic Data')
parser.add_argument('--data', type=str, default='/home/amadeu/anaconda3/envs/darts_env/cnn/data2/trainset.txt', help='location of the data corpus')

#parser.add_argument('--emsize', type=int, default=128,
#                    help='size of word embeddings')
#parser.add_argument('--nhid', type=int, default=128,
#                    help='number of hidden units per layer')
#parser.add_argument('--nhidlast', type=int, default=128,
#                    help='number of hidden units for the last rnn layer')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--init_channels', type=int, default=8, help='num of init channels') # args.C, args.num_classes, args.layers, args.steps=4, args.multiplier=4, args.stem_multiplier=3
parser.add_argument('--layers', type=int, default=3, help='total number of layers')
parser.add_argument('--num_classes', type=int, default=4, help='num of output classes') # args.C, args.num_classes, args.layers, args.steps=4, args.multiplier=4, args.stem_multiplier=3
parser.add_argument('--steps', type=int, default=4, help='total number of Nodes')
parser.add_argument('--multiplier', type=int, default=4, help='multiplier')
parser.add_argument('--stem_multiplier', type=int, default=3, help='stem multiplier')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=2,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=10,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.75,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.25,
                    help='dropout for hidden nodes in rnn layers (0 = no dropout)')
parser.add_argument('--dropoutx', type=float, default=0.75,
                    help='dropout for input nodes in rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.2,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--seed', type=int, default=3,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='EXP',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=0,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1e-3,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=5e-7,
                    help='weight decay applied to all weights')
parser.add_argument('--continue_train', action='store_true',
                    help='continue train from a checkpoint')
parser.add_argument('--small_batch_size', type=int, default=-1,
                    help='the batch size for computation. batch_size should be divisible by small_batch_size.\
                     In our implementation, we compute gradients with small_batch_size multiple times, and accumulate the gradients\
                     until batch_size is reached. An update step is then performed.')
parser.add_argument('--max_seq_len_delta', type=int, default=20,
                    help='max sequence length')
parser.add_argument('--single_gpu', default=True, action='store_false', 
                    help='use single GPU')
parser.add_argument('--gpu', type=int, default=0, help='GPU device to use')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_wdecay', type=float, default=1e-3,
                    help='weight decay for the architecture encoding alpha')
parser.add_argument('--arch_lr', type=float, default=3e-3,
                    help='learning rate for the architecture encoding alpha')
#parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
#parser.add_argument('--search', type=bool, default=True, help='which architecture to use')
#parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
#parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
args = parser.parse_args()


# import data
# '/home/amadeu/anaconda3/envs/darts_env/cnn/data2/trainset.txt'
# '/home/ascheppa/miniconda2/envs/darts/cnn/data2/trainset.txt'

train_object, valid_object, num_classes = dp.data_preprocessing(data_file =args.data, 
          seq_size = args.bptt, representation = 'onehot', model_type = 'CNN', batch_size=args.batch_size)


eval_batch_size = 10

_, valid_data, num_classes = dp.data_preprocessing(data_file =args.data, 
          seq_size = args.bptt, representation = 'onehot', model_type = 'CNN', batch_size=eval_batch_size)




if args.small_batch_size < 0:
    args.small_batch_size = args.batch_size
    
#if not args.continue_train:
#    args.save = 'search-{}'.format(args.save)
#    create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
    
args.save = '{}search-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
    

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

np.random.seed(args.seed)
torch.manual_seed(args.seed)


if args.continue_train:
    model = torch.load(os.path.join(args.save, 'model.pt'))
else:
    model = model.RNNModelSearch(args.bptt,# args.emsize, args.nhid, args.nhidlast,
                           args.dropout, args.dropouth, args.dropoutx, args.dropouti, args.dropoute, 
                           args.init_channels, args.num_classes, args.layers, args.steps, args.multiplier, args.stem_multiplier) 
   
    

size = 0
for p in model.parameters():
    size += p.nelement()
logging.info('param size: {}'.format(size))
logging.info('initial genotype:')

if args.cuda:
    if args.single_gpu:
        parallel_model = model.to(device)

    else:
        parallel_model = nn.DataParallel(model, dim=1).to(device)

else:
    parallel_model = model.to(device)
    
logging.info(model.genotype())
architect = Architect(parallel_model, args)
    

total_params = sum(x.data.nelement() for x in model.parameters())
logging.info('Args: {}'.format(args))
logging.info('Model total parameters: {}'.format(total_params))

# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000

if args.continue_train:
    optimizer_state = torch.load(os.path.join(args.save, 'optimizer.pt'))
    if 't0' in optimizer_state['param_groups'][0]:
        optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    optimizer.load_state_dict(optimizer_state)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)


def evaluate(data_source, batch_size=10):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
  
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(batch_size)
    
    for step, (input, target) in enumerate(valid_data):

        data, targets = input, target
        
        targets = torch.max(targets, 1)[1]


        log_prob, hidden = parallel_model(data, hidden)
        criterion = nn.CrossEntropyLoss()

        loss = criterion(log_prob, targets)
        
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 2))
        n = input.size(0)
        
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top2.update(prec5.data, n)
        
        if step % args.log_interval == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top2.avg)
        hidden = repackage_hidden(hidden)
        
    return top1.avg, objs.avg


def train(epoch):
    #assert args.batch_size % args.small_batch_size == 0, 'batch_size must be divisible by small_batch_size'

    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    # Turn on training mode which enables dropout.
    total_loss = 0
    start_time = time.time()
    hidden = [model.init_hidden(args.small_batch_size) for _ in range(args.batch_size // args.small_batch_size)]
    hidden_valid = [model.init_hidden(args.small_batch_size) for _ in range(args.batch_size // args.small_batch_size)]
  
    s_id = 0
    
    for step, (input, target) in enumerate(train_object):
        
        data, targets = input, target
        data_valid, targets_valid = next(iter(valid_object))
        
        targets = torch.max(targets, 1)[1]
        targets_valid = torch.max(targets_valid, 1)[1]
        
        
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
     
        seq_len = int(bptt)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 
       
        model.train()

        optimizer.zero_grad()

  
        cur_data, cur_targets = data, targets
        cur_data_valid, cur_targets_valid = data_valid, targets_valid

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden[s_id] = repackage_hidden(hidden[s_id])
        hidden_valid[s_id] = repackage_hidden(hidden_valid[s_id])

        hidden_valid[s_id], grad_norm = architect.step(
                hidden[s_id], cur_data, cur_targets,
                hidden_valid[s_id], cur_data_valid, cur_targets_valid,
                optimizer,
                args.unrolled)

        # assuming small_batch_size = batch_size so we don't accumulate gradients
        optimizer.zero_grad()
        
        hidden[s_id] = repackage_hidden(hidden[s_id])
        
        log_prob, hidden[s_id], rnn_hs, dropped_rnn_hs = parallel_model(cur_data, hidden[s_id], return_h=True)
        criterion = nn.CrossEntropyLoss()

        raw_loss = criterion(log_prob, cur_targets)

        loss = raw_loss
        # Activiation Regularization
        #if args.alpha > 0: # per default args.alpha=0, so we don't need this part
        #  loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss *= args.small_batch_size / args.batch_size
        total_loss += raw_loss.data * args.small_batch_size / args.batch_size
        
        loss.backward()

        gc.collect()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
     
        
        prec1,prec2 = utils.accuracy(log_prob, targets, topk=(1,2)) 
        
        n = input.size(0)
        
        objs.update(loss.data, n)
       
        top1.update(prec1.data, n) 

        # total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        
        if step % args.log_interval == 0 and step > 0:
            #logging.info(parallel_model.genotype())
            #logging.info(F.softmax(parallel_model.weights, dim=-1))
            logging.info('| step {:3d} | train obj {:5.2f} | '
                'train acc {:8.2f}'.format(step,
                                           objs.avg, top1.avg))

    logging.info('{} epoch done   '.format(epoch) + str(parallel_model.genotype()))

    return top1.avg, objs.avg





for epoch in range(1, args.epochs+1):
    # epoch=1
    epoch_start_time = time.time()
    
    train_acc, train_obj = train(epoch)
    
    # determine the validation loss in every 5th epoch 
    if epoch % 5 == 0:
        
        val_acc, val_obj = evaluate(val_data, eval_batch_size)
        logging.info('-' * 89)
        logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid acc {:5.2f} | '
                'valid obj {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_acc, val_obj))
        logging.info('-' * 89)

        #if val_loss < stored_loss:
        #    save_checkpoint(model, optimizer, epoch, args.save)
        #    logging.info('Saving Normal!')
        #    stored_loss = val_loss
    
        #best_val_loss.append(val_loss)
