f#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 16:11:24 2020

@author: amadeu
"""

import pandas as pd
import numpy as np
import sklearn
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import math
import torch



## BONAS results

import pickle


with open('/home/amadeu/trained_results/supermodel_random_100/trained_models.pkl', 'rb') as f:
    results = pickle.load(f)
    
res = results[0]["genotype"]



# get accuracy per number of samples


accs=[]
number_of_samples = []
genotypes = []
s=1
for result in results:
    accs.append(result["metrics"]) # y-axis
    number_of_samples.append(s)
    genotypes.append(result["genotype"])
    s+=1
    
    
gene = genotypes[0]
gene2 = genotypes[5]

concat = range(2,6)
gene2[1] = concat

genotype = [gene2[0], concat, gene2[2], concat, gene2[4], gene2[5]] # so sollte ich es jetzt an trainfinalArch übergeben können


# die listen hätten dann 1000 elemente. müsste dann noch den mean von 0:100, 100:200, ..., 900:1000 bilden (bzw. boxplot dazu) 
    


# get final architecture, to evaluate this final arch

max_acc=0
for i in range(len(results)):
    # i = 0
    # if-schleife nur True wenn neues bestes model
    if accs[i] > max_acc: # max_acc wird als 0 initialisiert, aber jetzt überschrieben mit neuem max_acc
        max_acc = accs[i] # neue max_acc wird überschrieben -> bestes model 
        genotype = genotypes[i] # genotype vom besten model wird überschrieben

# save genotype as np.array, same as in DARTS  


## DARTS results

# genotype_file ='/home/amadeu/anaconda3/envs/EXPsearch-try-20210620-144357-pdarts_geno.npy'
# genotype = np.load(genotype_file, allow_pickle=True)




train_loss = np.load('/home/amadeu/anaconda3/envs/GenomNet_MA/model_results/dilconv_sepconvs/stateless-train_loss-20210430-1616.npy')
valid_loss = np.load('/home/amadeu/anaconda3/envs/GenomNet_MA/model_results/2normalconvs/stateless-valid_loss-20210430-1842.npy')


import matplotlib.pyplot as plt
plt.plot(valid_loss, label='valid_loss')
plt.title('Cross-Entropy Loss')
#plt.ylim(1.36, 1.39)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


### Precision Recall, Confusion matrix

# acc_train = np.load('/home/amadeu/anaconda3/envs/GenomNet_MA/model_results/2normalconvs/stateless-acc_train-20210430-1842.npy')
# acc_valid = np.load('/home/amadeu/anaconda3/envs/GenomNet_MA/model_results/2normalconvs/stateless-acc_valid-20210430-1842.npy')



acc_train = np.load('/home/amadeu/anaconda3/envs/GenomNet_MA/fin-archsearch-20210516-143441-acc_train-20210516-2038.npy')
acc_valid = np.load('/home/amadeu/anaconda3/envs/GenomNet_MA/fin-archsearch-20210516-143441-acc_valid-20210516-2038.npy')

#acc_val = np.zeros(15)
#acc_val[0:5] = acc_valid[0]
#acc_val[5:10] = acc_valid[1]
#acc_val[10:15] = acc_valid[2]

plt.plot(acc_train, label='acc_train')
plt.plot(4,0.336,'ro', label = 'acc_val') 
plt.plot(9,0.3666,'ro') 
plt.plot(14,0.3683,'ro') 
#plt.ylim(1.36, 1.39)
plt.legend(bbox_to_anchor=(1.05, 1), loc='lower right', borderaxespad=0.)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.savefig("plot", dpi= 300)






# confusion-matrix & precision_recall
cm = confusion_matrix(y_pred, y_true)
precRec = classification_report(y_pred, y_true, output_dict=True)
precRec['accuracy']


# train_accuracy per epoch
train_acc = np.load('acc_train.npy') 
test_acc = np.load('acc_test.npy')



import pandas as pd
import numpy as np
import sklearn
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import math
import torch


random_search_results = np.load('/home/amadeu/anaconda3/envs/GenomNet_MA/genomicsRandomSearch/EXPsearch-try-20210606-120323-randomSearchResults.npy', allow_pickle=True)

# from 0:15 epoch_acc from stage1; from 15:30 epoch_acc from stage2; from 30:45 epoch_acc from stage3
acc_train_PDARTS = np.load('/home/amadeu/anaconda3/envs/EXPsearch-try-20210620-110508-acc_train-20210620-1109.npy', allow_pickle=True)
