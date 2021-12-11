

import copy
import torch.nn.functional as F
import torch.nn as nn
import torch
import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# kann von beliebigen Wert den durchschnitt berechnen
class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1): # val ist irgend ein wert, n ist anzahl an werten
        self.val = val
        self.sum += val * n # summiert den wert multipliziert mit seiner anzahl dazu
        self.count += n # zählt anzahl an werten
        self.avg = self.sum / self.count # bildet den durchschnitt
        
        

# model, data_loader, mask = supernet, train_loader_super, supernet_mask
def set_running_statistics(model, data_loader, mask):
    # Resetting statistics of subnets' BN, forwarding part of training data through each subnet to be evaluated
    bn_mean = {}
    bn_var = {}
    forward_model = copy.deepcopy(model).to(device)
    # names, ms = [],[]
    for name, m in forward_model.named_modules(): 
        # print(name) # cells.5._ops.13
        # print(m) # Sequential(
        #                       (0) MaxPool1d
        #                       (1) BatchNorm1d)
        # names.append(name) # also die einzelnen operationen/edges und nodes von meinem supernet
        # ms.append(m)
        # m = ms[3]
        # name = names[3]
        # mean_est = bn_mean[name]
        # var_est = bn_var[name]
        
        if isinstance(m, nn.BatchNorm1d): # nur für opertionen m, welche BatchNorm2d sind/enthalten

            bn_mean[name] = AverageMeter() 
            bn_var[name] = AverageMeter()

            # bn, mean_est, var_est = m, bn_mean[name], bn_var[name]
            def new_forward(bn, mean_est, var_est):
                
                def lambda_forward(x): # wird iwie an forward_model (was einfach nur model ist) angehängt und unten bei for i in enumerate(dataloader) ausgeführt, wo input
                # eben eingesetzt wird; deswegen ist das "x" hier einfach unser input
                    # x = input
                    # x = torch.rand(2,10,32,32)
                    
                    if x.dim() == 3: # for CNN operations
                        # print(x.shape)
                        batch_mean = x.mean(0, keepdim=True).mean(2, keepdim=True)#.mean(3, keepdim=True)  # damit shape [1, C, 1, 1] bei ihm! bei mir [1,C,1] also mean über batches und neuronen
                        # channelwise mean
                        batch_var = (x - batch_mean) * (x - batch_mean)
                        batch_var = batch_var.mean(0, keepdim=True).mean(2, keepdim=True)#.mean(3, keepdim=True)
    
                        batch_mean = torch.squeeze(batch_mean)
                        batch_var = torch.squeeze(batch_var)
    
                        mean_est.update(batch_mean.data, x.size(0)) # x.size(0) ist batch_size von input
                        var_est.update(batch_var.data, x.size(0))
    
                        # bn forward using calculated mean & var
                        _feature_dim = batch_mean.size(0) # num_channels von input
                    else: # for RHN operations
                        batch_mean = x.mean(0, keepdim=True)#.mean(3, keepdim=True)  # damit shape [1, C, 1, 1] bei ihm! bei mir [1,C,1] also mean über batches und neuronen
                        batch_var = (x - batch_mean) * (x - batch_mean)
                        batch_var = batch_var.mean(0, keepdim=True)#.mean(3, keepdim=True)
    
                        batch_mean = torch.squeeze(batch_mean)
                        batch_var = torch.squeeze(batch_var)
    
                        mean_est.update(batch_mean.data, x.size(0)) # x.size(0) ist batch_size von input
                        # macht "sum += val * n" in AverageMeter()
                        # sum = 0
                        # sum += batch_mean.data * x.size(0)
                        
                        var_est.update(batch_var.data, x.size(0))
    
                        # bn forward using calculated mean & var
                        _feature_dim = batch_mean.size(0) # num_chann
                        
                    if bn.affine:
                        weight = bn.weight[:_feature_dim].to(x.device)
                        bias = bn.bias[:_feature_dim].to(x.device)
                        return F.batch_norm(
                            x, batch_mean, batch_var, weight,
                            bias, False, 0.0, bn.eps,
                        )
                    else:
                        return F.batch_norm(
                            x, batch_mean, batch_var, None,
                            None, False, 0.0, bn.eps,
                        )

                return lambda_forward

            m.forward = new_forward(m, bn_mean[name], bn_var[name])

    batchsize = -1
    inputsize = 1024
    input_cat = [] # hier werden die ganzen inputs gespeichert
    # er iteriert jetzt ganz normal durch data_loader und jedes mal wird aber aber auch bei lambda forward durchgegangen, d.h. bei batchnormalisations werden mean und variance davon berechnet
    for i, (input, target) in enumerate(data_loader): # er iteriert jetzt erstmal durch und fügt jedesmal input zusammen bis wir ein riesen input haben mit batch_size 1024 (deswegen bei hidden initialisieren auch batchsize 1024)
        #print(input.shape)
        if batchsize == -1: # ist immer wahr weil hier paar zeilen drüber so definiert wurde
            batchsize = input.size(0) # batchsize, welches in train_loader_super definiert wurde
        if len(input_cat) * batchsize < inputsize: # für eine bestimmte anzahl wird input_cat mit input aufgefüllt (bis wir 1024 elemente in input_cat haben)
            input_cat.append(input)
            continue
        else:
            input = torch.cat(input_cat) # 511 elemente mit jeweils shape [2,4,1000] werden concated: -> [1022,4,1000]
            batch_size_new = len(input_cat) * batchsize
            input_cat = []
        with torch.no_grad():
            # input = input.transpose(1, 2).float()
            input = input.float().to(device)
           
            hidden = forward_model.init_hidden(batch_size_new) #inputsize*batchsize # forward_model ist einfach nur model kopiert, siehe oben bei set_running_statistics
            #print(input.shape)
            #print(hidden.shape)
            
            forward_model(input, hidden, mask) # supernet_mask
            
        if (i + 1) * batchsize > 10000:
        #if i > 2:
            break

    # jetzt werden alle parameters die batchnorm haben upgedated mit den 
    for name, m in model.named_modules():
        if name in bn_mean and bn_mean[name].count > 0:
            feature_dim = bn_mean[name].avg.size(0) # einfach nur num_channels von jeder operation (stem z.B. 24)
            assert isinstance(m, nn.BatchNorm1d)
            m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg) # running_mean von m wird upgedatet mit average
            m.running_var.data[:feature_dim].copy_(bn_var[name].avg)
