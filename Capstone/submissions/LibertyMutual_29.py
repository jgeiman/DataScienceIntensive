import time
import math

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

#from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier   
import xgboost as xgb
from sklearn.base import clone

#from sklearn.cross_validation import StratifiedKFold
#from sklearn.metrics import confusion_matrix, mean_squared_error
#from sklearn.grid_search import GridSearchCV

from gini import Gini
from preprocess import encode_categorical
from stacked import StackedClassiferRegressor 

def HazardBins(bin_id = 1):
    all_labels = ['low','medium','high','very high']
    exps = [math.exp(i) for i in range(6)]
    bins = None
    labels = None
    
    if bin_id == 1:
        bins = np.array([0, exps[1], exps[2], exps[3], exps[5]])
        labels = all_labels[:]
        
    elif bin_id == 2:
        bins = np.array([0, exps[2], exps[3], exps[5]])
        labels = all_labels[:3]

    elif bin_id == 3:
        bins = np.array([0, exps[1], exps[3], exps[5]])
        labels = all_labels[:3]
        
    return bins, labels    

    
train = pd.read_csv('./data/train.csv', index_col='Id')
test = pd.read_csv('./data/test.csv', index_col='Id')

columns = train.drop(['Hazard'], axis=1).columns

# encode categorical variables as numbers
train = encode_categorical(train)
test = encode_categorical(test)

code_id = 29
seed = 47

n = 100
#n_split = 50

gbm = xgb.XGBRegressor(max_depth=5, n_estimators=n, silent=False,
                       learning_rate=0.3, gamma = 0, seed=seed)

#reg = GradientBoostingRegressor(n_estimators = n, 
#                            min_samples_split = n_split, 
#                            learning_rate = 0.5,
#                            random_state = seed)

#regs = []
#for i in (10, 10, 10, 5):
#    r = clone(reg)
#    r.min_sample_split = i    
#    regs.append(r)
    
clf = KNeighborsClassifier(n_neighbors=5, weights='distance')

stacked = StackedClassiferRegressor(clf, gbm)

# get bins and labels
bins, labels = HazardBins(3)
#labels = range(len(bins)-1)

# trained the stacked classifier & regressor
start_time = time.time()
stacked.fit(train[columns].as_matrix(), train['Hazard'].as_matrix(), bins, labels)
stop_time = time.time()
print "training time: %.2fs" % (stop_time - start_time)

haz_pred = stacked.predict(train[columns].as_matrix())
train_gini = Gini(haz_pred, train.Hazard)
print 'Gini (training):', train_gini

plt.scatter(train.Hazard, haz_pred)

# output predictions for submission
result = pd.DataFrame({'id': test.index})
result['Hazard'] = stacked.predict(test[columns].as_matrix())
result.to_csv('%d_stacked_result.csv' % code_id, index=False, sep=',')
