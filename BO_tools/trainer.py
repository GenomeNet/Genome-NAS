#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 17:21:29 2021

@author: amadeu
"""

import numpy as np
import torch
import opendomain_utils.training_utils as utils
import logging
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import random
from BONAS_search_space.super_model import geno2mask, merge
from BONAS_search_space.model_search import RNNModelSearch

from opendomain_utils.ioutils import copy_log_dir
from opendomain_utils.genotypes import Genotype
from BO_tools.configure_files import distributed

DataParallel = torch.nn.parallel.DistributedDataParallel if distributed else torch.nn.DataParallel

from itertools import cycle, islice

from opendomain_utils.bn_utils import set_running_statistics
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import generalNAS_tools.data_preprocessing_new as dp



 #   train_supernet_epochs=args.train_supernet_epochs
 #   data_path=os.path.join(local_data_dir, 'data')
 #   super_batch_size=args.super_batch_size#64,
 #   sub_batch_size=args.sub_batch_size#128,
    
 #   cnn_lr=args.cnn_lr
 #   cnn_weight_decay=args.cnn_weight_decay
 #   rhn_lr=args.rhn_lr
 #   rhn_weight_decay=args.rhn_weight_decay
 #   momentum=args.momentum
    
 #   report_freq=args.report_freq
 #   epochs=args.epochs
 #   init_channels=args.init_channels#36,
 #   layers= args.layers#20,
 #   drop_path_prob=args.drop_path_prob
 #   seed=0
 #   grad_clip=args.clip
 #   parallel=False
 #   mode=args.mode
 #   train_directory = args.train_directory
 #   valid_directory = args.valid_directory
 #   num_files = args.num_files
 #   seq_len = args.seq_len
 #   num_steps = args.num_steps
 #   next_character_prediction=args.next_character_prediction
 #   dropouth = args.dropouth
 #   dropoutx = args.dropoutx
 #   one_clip = args.one_clip
 #   clip = args.clip
 #   conv_clip = args.conv_clip
 #   rhn_clip = args.rhn_clip 








class Trainer:
    def __init__(self,
                 train_supernet_epochs=5,
                 data_path='data',
                 super_batch_size=64,
                 sub_batch_size=128,
                 
                 cnn_lr=0.025,
                 cnn_weight_decay=3e-4,
                 rhn_lr=2,
                 rhn_weight_decay=5e-7,
                 momentum=0.9,
                 
                 report_freq=2,
                 num_steps=2,
                 epochs=50,
                 init_channels=8,
                 layers=3,
                 drop_path_prob=0.2,
                 seed=0,
                 parallel = False,
                 mode='uniform',
                 train_directory = '/home/amadeu/Downloads/genomicData/train',
                 valid_directory = '/home/amadeu/Downloads/genomicData/validation',
                 num_files = 3,
                 seq_len = 120,
                 dropouth = 0.25,
                 dropoutx = 0.75,
                 next_character_prediction=True,
                 one_clip=True,
                 clip=5,
                 conv_clip=5,
                 rhn_clip=0.25,
                 steps = 4,
                 multiplier = 4, 
                 stem_multiplier = 3 
                 ):
        
        self.parallel = parallel
        self.train_supernet_epochs = train_supernet_epochs
        self.data_path = data_path
        self.super_batch_size = super_batch_size
        self.sub_batch_size = sub_batch_size
        
        self.cnn_lr = cnn_lr
        self.cnn_weight_decay = cnn_weight_decay
        self.rhn_lr = rhn_lr
        self.rhn_weight_decay = rhn_weight_decay
        self.momentum = momentum
        
        self.mode = mode
        
        self.report_freq = report_freq
        self.epochs = epochs
        self.init_channels = init_channels
        self.layers = layers
        self.drop_path_prob = drop_path_prob
        self.seed = seed
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.train_directory = train_directory
        self.valid_directory = valid_directory
        self.num_files = num_files
        self.seq_len = seq_len
        self.num_steps = num_steps
        self.dropouth = dropouth
        self.dropoutx = dropoutx
        self.steps = steps
        self.multiplier = multiplier
        self.stem_multiplier = stem_multiplier
        self.next_character_prediction = next_character_prediction
        self.one_clip = one_clip
        self.clip = clip
        self.conv_clip=conv_clip
        self.rhn_clip=rhn_clip

        self.build_dataloader()
    

        self.train_loader_super, self.train_loader_sub, self.valid_loader, self.num_classes = self.build_dataloader()
        # train_loader_super, train_loader_sub, valid_loader = build_dataloader()



    
    
    # only difference between train_loader_sub and train_loader_super is different batch_size, and one is for the supermodel and the
    # other for the subnets
    
    
    def build_dataloader(self):
        train_loader_sub, valid_loader, num_classes = dp.data_preprocessing(self.train_directory, self.valid_directory, self.num_files, self.seq_len, self.sub_batch_size, self.next_character_prediction)
        
        train_loader_super, valid_queue, num_classes = dp.data_preprocessing(self.train_directory, self.valid_directory, self.num_files, self.seq_len, self.super_batch_size, self.next_character_prediction)
        
        return train_loader_super, train_loader_sub, valid_loader, num_classes
    
     #def build_dataloader():
        # train_loader_sub, valid_loader, num_classes = dp.data_preprocessing(train_directory = train_directory, valid_directory = valid_directory, num_files= num_files,
        #   seq_size = seq_len, batch_size= sub_batch_size, next_character=next_character_prediction)
        
        # train_loader_super, valid_queue, num_classes = dp.data_preprocessing(train_directory = train_directory, valid_directory = valid_directory, num_files= num_files,
        #   seq_size = seq_len, batch_size= sub_batch_size, next_character=next_character_prediction)
        
     #   return train_loader_super, train_loader_sub, valid_loader
    

    def set_seed(self):
        np.random.seed(self.seed)
        cudnn.benchmark = True
        torch.manual_seed(self.seed)
        cudnn.enabled = True
        torch.cuda.manual_seed(self.seed)



    def build_model(self, mask):
        model = RNNModelSearch(self.seq_len, self.dropouth, self.dropoutx,
                        self.init_channels, self.num_classes, self.layers, self.steps, self.multiplier,
                        self.stem_multiplier, True, 0.2, None, mask)
        #if self.parallel:
        #    if distributed:
        #        model = DataParallel(model.cuda(), device_ids=[torch.cuda.current_device()])
        #    else:
        #        model = DataParallel(model.cuda())
        #else:
        #    model = model.cuda()
        return model
    
    
    
    #seq_len, dropout, dropouth, dropoutx, dropouti, dropoute, init_channels, num_classes, layers, steps, multiplier, stem_multiplier = 10, 0.75, 0.25, 0.75, 0.2, 0, 16, 4, 3, 4, 4, 3
    #criterion = nn.CrossEntropyLoss().to(device)
    
    #def build_model(mask):
    #    model = RNNModelSearch(seq_len, dropouth, dropoutx,
    #                    init_channels, num_classes, layers, steps, multiplier,
    #                    stem_multiplier, True, 0.2, None,
    #                    mask)
        #if self.parallel:
        #    if distributed:
        #        model = DataParallel(model.cuda(), device_ids=[torch.cuda.current_device()])
        #    else:
        #        model = DataParallel(model.cuda())
        #else:
        #    model = model.cuda()
    #    return model
    
    #model = RNNModelSearch(seq_len, dropout, dropouth, dropoutx, dropouti, dropoute,
    #                    init_channels, num_classes, layers, steps, multiplier,
    #                    stem_multiplier, True, 0.2, None,
    #                    mask)
    
    #def build_model(mask):
    #    model = Network(init_channels, CIFAR_CLASSES, layers, mask=mask)
    #    if parallel:
    #        if distributed:
    #            model = DataParallel(model.cuda(), device_ids=[torch.cuda.current_device()])
    #        else:
    #            model = DataParallel(model.cuda())
    #    else:
    #        model = model.cuda()
    #    return model


    # archs, eval_archs = genotypes, eval_genotypes
    def train_and_eval(self, archs, eval_archs=None):
        """
        :param archs: archs sample by parallel BO
        :return: results list<genotype, top1acc>
        """
        # archs, eval_archs = genotypes, eval_genotypes
        self.genotypes = [eval(arch) if isinstance(arch, str) else arch for arch in archs] # speichert die genotypes jetzt als liste 
        self.eval_genos = None
        # ist None also überspringen
        if eval_archs != None:
            self.eval_genos = [eval(arch) if isinstance(arch, str) else arch for arch in eval_archs]
            self.genotypes = self.genotypes + self.eval_genos
        self.subnet_masks = [geno2mask(genotype) for genotype in self.genotypes]
        # genotype = genotypes[0]
        # subnet_masks = [geno2mask(genotype) for genotype in genotypes]
        # subnets_cnn = [for subnet[0] in subnets]
        cnn_masks = []
        rnn_masks = []
        for cnn_sub in self.subnet_masks:
            cnn_masks.append(cnn_sub[0])
        for rnn_sub in self.subnet_masks:
            rnn_masks.append(rnn_sub[1])
        
        # supernet_mask = merge(subnet_cnn_masks, subnet_rnn_masks)

        self.supernet_mask = merge(cnn_masks, rnn_masks) # die 5 subnets zusammengefügt, damit er 1 großes supernet hat, welches aus den 100 init_samples/subarchitecturen gebildet wurde
        
        # len(subnet_masks)=5 und len(train_loader_sub)=391: erzeugt also eine list mit 391 elementen die eben immer 0,1,2,3,4 (wegen len(subnet_masks) sind)
        self.iterative_indices = list(islice(cycle(list(range(len(self.subnet_masks)))), len(self.train_loader_sub))) 
        # train_loader_sub wird als build_dataloader initialisiert und diese ist eine funktion 2 weiter oben definiert 
        # iterative_indices = list(islice(cycle(list(range(len(subnet_masks)))), len(train_loader_sub)))
        
       
        supernet = self.build_model(self.supernet_mask) # baut das model, gemäß der mask
        # supernet = build_model(supernet_mask) # baut das model, gemäß der mask

        # am ende haben wir ähnliches one-shot model wie in 
        # init_channels, layers = 36,20
        # supernet = Network(init_channels, CIFAR_CLASSES, layers, mask=supernet_mask)

        logging.info("Training Super Model ...")
        logging.info("param size = %fMB", utils.count_parameters_in_MB(supernet))
        
        
        conv = []
        rhn = []
        for name, param in supernet.named_parameters():
            #print(name)
            #if 'stem' or 'preprocess' or 'conv' or 'bn' or 'fc' in name:
           if 'rnns' in name:
               #print(name)
               rhn.append(param)
           #elif 'decoder' in name:
           else:
               #print(name)
               conv.append(param)
        self.conv = conv
        self.rhn = rhn
        
        optimizer = torch.optim.SGD([{'params':conv}, {'params':rhn}], lr=self.cnn_lr, weight_decay = self.cnn_weight_decay)
        optimizer.param_groups[0]['lr'] = self.cnn_lr
        optimizer.param_groups[0]['weight_decay'] = self.cnn_weight_decay
        optimizer.param_groups[0]['momentum'] = self.momentum
        optimizer.param_groups[1]['lr'] = self.rhn_lr
        optimizer.param_groups[1]['weight_decay'] = self.rhn_weight_decay
        
        
        #optimizer = torch.optim.SGD([{'params':conv}, {'params':rhn}], lr=cnn_lr, weight_decay = cnn_weight_decay)
        #optimizer.param_groups[0]['lr'] = cnn_lr
        #optimizer.param_groups[0]['weight_decay'] = cnn_weight_decay
        #optimizer.param_groups[0]['momentum'] = momentum
        #optimizer.param_groups[1]['lr'] = rhn_lr
        #optimizer.param_groups[1]['weight_decay'] = rhn_weight_decay
        
        
        #optimizer = torch.optim.SGD(
        #    supernet.parameters(), # er nimmt also nur die parameters vom "kleinen supernet" welches nur 100 subarchitectures beinhaltet
        #    self.learning_rate,
        #    momentum=self.momentum,
        #    weight_decay=self.weight_decay
        #)
        
      
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(self.epochs))
    
        #params3 = []
        #for param in supernet.named_parameters():
        #    params3.append(param)

        # jetzt wird das modell erstmal für viele/1ne epochen mit den init_samples trainiert, um die gewichte des NN zu fitten
        # er trainiert erstmal für 1ne Epoche das supernetwork, indem er für jeden step ein child model sampled (haben 3000 steps also sampled er 3000 mal aus den 100/subnets)
        # danach validiert er dann mit dem supernetwork
        # und dann um ergebnisse zu bekommen für GCN, evaluiert er, indem er die 100/5 subnets jeweils für 3000 steps evaluiert
        for epoch in range(self.epochs): 
            # epoch=0
            logging.info('epoch %d lr %e', epoch, scheduler.get_last_lr()[0])
            supernet.drop_path_prob = self.drop_path_prob * epoch / self.epochs
            # supernet.drop_path_prob = drop_path_prob * epoch / epochs

            if epoch in range(self.train_supernet_epochs): 
                # train supernet, which contains all subnets 
                self.train_supernet(supernet, optimizer, epoch)
                
            else:
                self.train(supernet, optimizer, supernet=False)
               
                
            scheduler.step()
        logging.info("Evaluating subnets ...")
        
        # evaluate subnets with the weights trained with the supernet
        results = self.evaluate_subnets(supernet, self.subnet_masks, self.genotypes, self.eval_genos) # supernet ist das model, subnet_masks sind die 100/5 subnets einzeln abgespeichert

        return results

    def train_supernet(self, model, optimizer, epoch):
        self.train(model, optimizer, supernet=True) # train supernet for one epoch
        if epoch == self.train_supernet_epochs - 1: 
            val_obj, val_top1, val_top2 = self.evaluate(model, self.supernet_mask) # evaluate supernet 
            logging.info('Supernet valid %e %f %f', val_obj, val_top1, val_top2)
            # update batchnorm operations with running mean: he resets the bn's
            set_running_statistics(model, train_loader_super, supernet_mask)
            val_obj, val_top1, val_top2 = self.evaluate(model, self.supernet_mask) # evaluate supernet after bn resets
            logging.info('After resetbn Supernet valid %e %f %f', val_obj, val_top1, val_top5)

        copy_log_dir()

    def evaluate_subnets(self, supernet, subnet_masks, genotypes, eval_genos=None):
        results = []
        if eval_genos:
            genotypes = eval_genos
            subnet_masks = [geno2mask(geno) for geno in genotypes]
        with torch.no_grad():
            supernet.eval()
            supernet_copy = copy.deepcopy(supernet)
            i=1
            # masks, genotypess = [], []
            # s=0
            # top1 = [0.1,0.3,0.5] #0.2, 0.6]
            # er iteriert jetzt durch die einzelnen child models/den subnets (von welchen es 5/10 gibt), also evaluiert jedes einzelne
            for mask, genotype in zip(subnet_masks, genotypes): # er macht 5 iterationen für die 5 genotypes eben
                # results.append((genotype, top1[s])) # damit ich eine results liste bekomme
                # s+=1
                
                set_running_statistics(supernet_copy, self.train_loader_sub, mask) # für mein search_space muss ich glaube ich nichts ändern, weil
                # es ja nur BatchNorm betrifft und diese dann bei RHN's gar nicht aktiviert werden
                obj, top1, top5 = self.evaluate(supernet_copy, mask) 
                logging.info('%s th Arch %s valid %e %f %f',str(i), str(genotype.normal), obj, top1, top5)
                # 
                results.append((genotype, top1))
                copy_log_dir()
                i+=1
        return results
    
    def evaluate(self, model, mask):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top2 = utils.AvgrageMeter()
        model.eval()
        
        with torch.no_grad():
            
            for step, (input, target) in enumerate(self.valid_loader):
                
                input = Variable(input).to(device)#.cuda()
                target = Variable(target).to(device)#.cuda(async=True)
                
                if step > self.num_steps:
                    break
                #input, target = input, target
               
                #input = input.transpose(1, 2).float()
        
                target = torch.max(target, 1)[1]
                batch_size = input.size(0)
    
                hidden = model.init_hidden(batch_size)
                
                logits, hidden, rnn_hs, dropped_rnn_hs  = model(input, hidden, mask)
                
                loss = self.criterion(logits, target)
                prec1, prec2 = utils.accuracy(logits, target, topk=(1, 2))
                objs.update(loss.data.item(), batch_size)
                top1.update(prec1.data.item(), batch_size)
                top2.update(prec2.data.item(), batch_size)
        return objs.avg, top1.avg / 100, top2.avg / 100

    # model, supernet = supernet, True
    # The super-network is trained by uniformly/randomly sampling from the
    # architectures of supernets. In each iteration, one sub-network (Ai, Xi) is randomly sampled from the super-network,
    def train(self, model, optimizer, supernet=False):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top2 = utils.AvgrageMeter()
        # model = supernet
        model.train()
        iterative_indices = np.random.permutation(self.iterative_indices)
        train_loader = self.train_loader_super if supernet else self.train_loader_sub
        # train_loader = train_loader_super if supernet else train_loader_sub
        
        #params2 = []
        #for param in model.named_parameters():
        #    params2.append(param)

        for step, (input, target) in enumerate(train_loader):
            
            input = Variable(input.float()).to(device)#.cuda()
            target = Variable(target).to(device)#.
            
            if step > self.num_steps:
                break
            #input, target = input, target
           
            #input = input.transpose(1, 2).float()
    
            target = torch.max(target, 1)[1]
            batch_size = input.size(0)

                
            hidden = model.init_hidden(batch_size)
            #hidden_valid = model.init_hidden(args.batch_size
            
            if self.mode == 'uniform': # in trainer_config wurde mode='random' übergeben
                mask = self.subnet_masks[iterative_indices[step]] if not supernet else self.supernet_mask
            else:
                # wir haben zum einen "self.supernet_mask" wo alle 5 subnets zusammengemerged wurden
                # und wir haben "self.subnet_masks" wo die 5 subnets (welche jeweils aus mask_cnn und mask_rnn besteht) noch einzeln gespeichert sind
                mask = random.choice(self.subnet_masks) # subnet_mask enthält ja die adjacency matrizen zu 5 modellen
                # er sampled hier 1 aus diesen 5 raus und diesen setzt er dann in model ein!!
                # er sampled aber in jedem step ein neues aus diesen 5 raus, d.h. bei 3000 steps hätten wir 3000 mal child_model gesampled
            #input = Variable(input).to(device)#.cuda()
            #target = Variable(target).to(device)#.cuda(async=True)
            
            optimizer.zero_grad()
            
            logits, hidden, rnn_hs, dropped_rnn_hs  = model(input, hidden, mask)
            
            # criterion = nn.CrossEntropyLoss()
            loss = self.criterion(logits, target)
            loss.backward()
            
            if self.one_clip == True:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip)
            else:
                torch.nn.utils.clip_grad_norm_(self.conv, self.conv_clip)
                torch.nn.utils.clip_grad_norm_(self.rhn, self.rhn_clip)
            
            optimizer.step()

            prec1, prec2 = utils.accuracy(logits, target, topk=(1, 2))
            objs.update(loss.data.item(), batch_size)
            top1.update(prec1.data.item(), batch_size)
            #top2.update(prec2.data.item(), batch_size)
            if step % self.report_freq == 0:
                #logging.info('supernet train %03d %e %f %f', step, objs.avg, top1.avg / 100, top2.avg / 100)
                logging.info('| step {:3d} | train obj {:5.2f} | '
                'train acc {:8.2f}'.format(step,
                                           objs.avg, top1.avg))
                copy_log_dir()
                
                
