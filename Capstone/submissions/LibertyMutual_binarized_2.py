# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 15:29:21 2015

@author: jag
"""

# For each observation, set target = 1 if target >= x, 0 otherwise. 
# Do this for x = 2, 3, 4, ... up to a number that you see fit. This means 
# you have K number of classification problems, where K = the number of 
# different x's you've decided on. Finally, sum or ensemble the predictions 
# from the K classification problems for each test observation.

# For example, let's say that target could be from 1 to 10. Then you can turn 
# this into 9 binary classification problems, one for x >= 2, x >= 3, ... 
# x >= 10. When you take the predictions from the classification problems and 
# sum them together, then you'll get a value between 0 and 9, or 1 and 10 if 
# you decide to add 1 automatically to each observation. If indeed your 
# binary classifiers are perfect, then by summing together you've exactly 
# reproduced the original target value.

import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from gini import Gini
from preprocess import encode_categorical

# function used to binarize the hazards
def binarizer(x, threshold):
    return int(x > threshold)
     
train = pd.read_csv('./data/train.csv', index_col='Id')
test = pd.read_csv('./data/test.csv', index_col='Id')

columns = train.drop(['Hazard'], axis=1).columns

# encode categorical variables as numbers
train = encode_categorical(train)
test = encode_categorical(test)

code_id = 100

# train a random forest
n = 500
n_split = 4

low = np.arange(2,31)
high = np.arange(35,75,5)
haz_bins = np.concatenate((low, high))
weights = np.concatenate((np.tile([1], len(low)), np.tile([5], len(high))))

haz_pred_i = np.zeros((test.shape[0], haz_bins.shape[0]))
train_pred_i = np.zeros((train.shape[0], haz_bins.shape[0]))

start_time = time.time()
for i, (haz, weight) in enumerate(zip(haz_bins, weights)):
    print 'training hazard > %d, elapsed time: %.2fs' % \
            (haz, time.time() - start_time) 

    y = train['Hazard'].apply(binarizer, args=(haz,))        
    
    clf = RandomForestClassifier(n_estimators=n, 
                            min_samples_split = n_split, 
                            random_state=47)
    
    # fit the classifier on the binarized hazard
    clf.fit(train[columns], y)
    
    # predict the hazard for this bin
    haz_pred_i[:,i] = clf.predict(test[columns]) * weight
    train_pred_i[:,i] = clf.predict(train[columns]) * weight

# combine the hazard predictions for each bin        
haz_pred = haz_pred_i.sum(axis=1) + 1    
train_pred = train_pred_i.sum(axis=1) + 1    

stop_time = time.time()
print "training time: %.2fs" % (stop_time - start_time)

plt.scatter(train.Hazard, train_pred)
plt.plot((0,70),(0,70), 'k--')

## output predictions for submission
result = pd.DataFrame({'id': test.index})
result['Hazard'] = haz_pred
result.to_csv('%d_binarized_result.csv' % code_id, index=False, sep=',')