#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 10:34:43 2021

@author: amadeu
"""


import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable

import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from sklearn.metrics import f1_score, recall_score, precision_score


true1 = torch.zeros(4,4)
true1[3][2] = 0.0

pred1 = torch.zeros(4,4)
pred1[3][2] = 1.0


true2 = torch.zeros(4,4)
true2[3][3] = 1.0

pred2 = torch.zeros(4,4)
pred2[3][1] = 1.0


re = recall_score(y_true=true1, y_pred=pred1, average='weighted')




from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
import numpy as np

iris = datasets.load_iris()
X = iris.data
y = iris.target

# Add noisy features
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# Limit to the two first classes, and split into training and test
X_train, X_test, y_train, y_test = train_test_split(X[y < 2], y[y < 2],
                                                    test_size=.5,
                                                    random_state=random_state)

# Create a simple classifier
classifier = svm.LinearSVC(random_state=random_state)
classifier.fit(X_train, y_train)
y_score = classifier.decision_function(X_test)


from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_score)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))



from sklearn.preprocessing import label_binarize

# Use label_binarize to be multi-label like settings
Y = label_binarize(y, classes=[0, 1, 2])
n_classes = Y.shape[1]

# Split into training and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5,
                                                    random_state=random_state)

# We use OneVsRestClassifier for multi-label prediction
from sklearn.multiclass import OneVsRestClassifier

# Run classifier
classifier = OneVsRestClassifier(svm.LinearSVC(random_state=random_state))
classifier.fit(X_train, Y_train)
y_score = classifier.decision_function(X_test)

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score, average_recall_score

# For each class
precision = dict()
recall = dict()
average_precision = dict()
recall_classes = dict()
for i in range(n_classes):
    # i=0
    precision[i], recall[i], _ = precision_y_score[:, irecall_curve(Y_test[:, i],
                                                        y_score[:, i])
    
    average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])
    
    average_recall[i] = recall_score(Y_test[:, i], y_score[:, i])
    average_recall[i] = recall_score(true2[:, i], pred2[:, i])
    
    recall_score(true2, pred2, average='weighted')
    
    
    true_all = torch.cat((true1, true2), dim=0)
    pred_all = torch.cat((pred1, pred2), dim=0)
    
    recall_score(true_all, pred_all, average='weighted')



# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
    y_score.ravel())
average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))


#from sklearn.metrics import f1_score
#y_true = [0, 1, 2, 0, 1, 2]
#y_pred = [0, 2, 1, 0, 0, 1]

#y_true = np.array([[1,0,0,0], [1,1,0,0], [1,1,1,1]])
#y_pred = np.array([[1,0,0,0], [1,1,1,0], [1,1,1,1]])

#re = recall_score(y_true=y_true, y_pred=y_pred, average='weighted')
#pre = precision_score(y_true=y_true, y_pred=y_pred, average='weighted')
#f1 = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')