#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 12:15:44 2020

@author: amadeu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 20:40:46 2020

@author: amadeu
"""

import torch
from collections import Counter
import numpy as np
from numpy import array
from torch.utils.data import Dataset,DataLoader

#### with test_set ####
#test_text = pd.read_csv('/home/amadeu/Desktop/solarTimeSeries/solar_venv/testset.csv')#.rename(columns={'date':'timestamp'}).set_index('timestamp')

#train_set = test_text[:700]
#valid_set = test_text[700:999]

# take first,second,thirs etc. element/observation (here first observation with element 0)
#test1 = test_text.iloc[0] # first row
#word_input = test1.iloc[0] # feature from observation
#target = test1.iloc[1] # target from observation
        
#def split(word_input): 
#    return [char for char in word_input] 
    
#words = split(word_input) # split all values from the feature into single values
    
#feature, target = list(), list()
#for i in range(9): # d.h. 9 (9 sollte mit while ersetzt werden) sequenzen á 3er steps # am besten mit while schleife, solange bis eben letztes element von word (hier 1000ste), das ende von der sequenz ist (also i+3)
#   #i = 0
#    feat_vals = words[i:i+3]
#    seq = np.zeros(3, dtype = int)
#    for ii in range(3):
#        seq[ii] = vocab_to_int[feat_vals[ii]]
#        print('seq')
#        print(seq)
#    #print(feat_vals)
#    feature.append(seq)
#    #targ_vals = words[i+3]
#    #target.append(targ_vals)
#    #print(target)


#### with trainset ####

#train_file='trainset.txt'
#test_file = 'testset.csv'
#seq_size=3
#batch_size= 2
#embedding_size=4
#lstm_size=5
#gradients_norm=5

seq_size = 5
batch_size=32
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


def open_data(train_file):
    with open(train_file, 'r') as f:
        text = f.read()
    
    def split(text): 
        return [char for char in text] 
        
    text = split(text)
    
    text = text[:-1]
        
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    word_counts = Counter(text)# counter object, brauche ich nur für sorted_vocab bzw. int_to_vacab und vocab_to_int dann
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {k: w for k, w in enumerate(sorted_vocab)}
    vocab_to_int = {w: k for k, w in int_to_vocab.items()}
    n_vocab = len(int_to_vocab)
    
    int_text = [vocab_to_int[w] for w in text] 
    
    text = np.asarray(int_text)
    #texttes = texttes[:-1]#weil letzte zeile ist leerzeichen und diese klasse brauch ich nicht
    return text, n_vocab, int_to_vocab, vocab_to_int


def create_sequences(in_text, seq_size, model_type = 'CNN'): #model_type = 'CNN'):#sequence ist einfach nur der input text
    x = list()
    y = list()
    text_len = len(in_text)
    for i in range(text_len):
        
        idx = i + seq_size #sequenzende; bei erster iter: 0+3=3
        
        if idx > len(in_text)-1: #wenn am ende (als 2milionste element), dann höre auf
            break
        if (model_type == 'CNN'):
            feat_seq, target_seq = in_text[i:idx], in_text[idx] # für CNN
            x.append(feat_seq)
            y.append(target_seq)
        if (model_type == 'LSTM'):
            feat_seq, target_seq = in_text[i:idx], in_text[(i+1):(idx+1)] # für LSTM
            x.append(feat_seq)
            y.append(target_seq)
    return array(x), array(y)



class get_data(Dataset):
    def __init__(self,feature,target):
        self.feature = feature
        self.target = target
    def __len__(self):
        return len(self.feature)
    def __getitem__(self,idx):
        item = self.feature[idx]
        label = self.target[idx]
        return item,label
    
    

text, n_vocab, int_to_vocab, vocab_to_int = open_data('/home/scheppacha/data/trainset.txt')#'/home/amadeu/Desktop/genom_Models/genom_venv/data/trainset.txt')#'/home/scheppacha/data/trainset.txt')

x,y = create_sequences(text,seq_size, model_type = 'CNN')

train_x, train_y = x[0:1500000,:], y[0:1500000]

valid_x,valid_y = x[1500000:2000000,:], y[1500000:2000000]


#num_vals = len(train_x)
# if train_x ungerade, dann lösche letzte zeile
#if num_vals % 2: #ungrade
#    train_x = train_x[:-1, :]
#    train_y = train_y[:-1]


#num_vals = len(valid_x)
# if train_x ungerade, dann lösche letzte zeile
#if num_vals % 2: #ungrade
#    valid_x = valid_x[:-1, :]
#    valid_y = valid_y[:-1]

#train = get_data(train_x,train_y)# erstes element ist einfach nur erste sequenz, zweites element ist die normale zeitreihe, drittes element ist alle sequenzen mit den jeweils 3 timesteps
#valid = get_data(valid_x,valid_y)
#train_loader = torch.utils.data.DataLoader(train,batch_size=32,shuffle=True)# hier greift er auf train und valid von oben zurück, macht wrsl. so ein torch object, welches dann immer durchiterieren kann mit enumerate(trainLoader)
#valid_loader = torch.utils.data.DataLoader(valid,batch_size=32,shuffle=True)

num_values_batch = batch_size*seq_size
# train 
num_batches_train = np.prod(train_y.shape) // (seq_size * batch_size)# sind 9375; gilt nur für benchmark preds, also valid_loader nicht Train()
#bleibt_über = np.prod(train_y.shape) - num_batches*(32*5)
end_train = num_values_batch*num_batches_train

# valid
num_batches_valid = np.prod(valid_y.shape) // (seq_size * batch_size)#sind 3124;gilt nur für benchmark preds, also valid_loader nicht Train()
#bleibt_über = np.prod(valid_y.shape) - num_batches*(32*5)
end_valid = num_values_batch*num_batches_valid

train_x, train_y = train_x[0:end_train,:], train_y[0:end_train]
valid_x, valid_y = valid_x[0:end_valid,:], valid_y[0:end_valid]

train = get_data(train_x,train_y)# erstes element ist einfach nur erste sequenz, zweites element ist die normale zeitreihe, drittes element ist alle sequenzen mit den jeweils 3 timesteps
valid = get_data(valid_x, valid_y)
train_loader = torch.utils.data.DataLoader(train,batch_size=32,shuffle=True)# hier greift er auf train und valid von oben zurück, macht wrsl. so ein torch object, welches dann immer durchiterieren kann mit enumerate(trainLoader)
valid_loader = torch.utils.data.DataLoader(valid,batch_size=32,shuffle=True)  

