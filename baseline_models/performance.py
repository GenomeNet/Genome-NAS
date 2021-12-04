#!/usr/bin/env python3
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

valid_loss = np.load('valid_loss.npy')
train_loss = np.load('train_loss.npy')


import matplotlib.pyplot as plt
plt.plot(valid_loss,label='valid_loss')
plt.title('Cross-Entropy Loss')
#plt.ylim(1.36, 1.39)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


### Precision Recall, Confusion matrix

y_pred = np.load('preds.npy')
y_true = np.load('trues.npy')

# confusion-matrix & precision_recall
cm = confusion_matrix(y_pred, y_true)
precRec = classification_report(y_pred, y_true, output_dict=True)
precRec['accuracy']


# train_accuracy per epoch
train_acc = np.load('acc_train.npy') 
test_acc = np.load('acc_test.npy')


predictions = np.load('EXPsearch-try-20210715-194521-predictions_train-20210715-1945.npy')