import time
import math

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
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

code_id = 10
seed = 47

n = 500
n_split = 50

reg = RandomForestRegressor(n_estimators = n, 
                            min_samples_split = n_split, 
                            random_state = seed)

regs = []
for i in (50, 50, 10, 5):
    r = clone(reg)
    r.min_sample_split = i    
    regs.append(r)
    
clf = RandomForestClassifier(n_estimators = n, 
                            min_samples_split = 5, 
                            random_state = seed)

stacked = StackedClassiferRegressor(clf, regs = regs)

# get bins and labels
bins, labels = HazardBins(2)

# trained the stacked classifier & regressor
start_time = time.time()
stacked.fit(train[columns], train['Hazard'], bins, labels)
stop_time = time.time()
print "training time: %.2fs" % (stop_time - start_time)

haz_pred = stacked.predict(train[columns])
train_gini = Gini(haz_pred, train.Hazard)
print 'Gini (training):', train_gini
#print 'Importances: ', clf.feature_importances_

#train['predicted'] = haz_pred
#train['haz_class'] = stacked._classes
#g = sns.factorplot(x="haz_class", y="predicted", data=train)
#g = sns.FacetGrid(train, col="haz_class")
#g = g.map(plt.hist, "predicted")

plt.scatter(train.Hazard, haz_pred)

## plot importances
#importances = clf.feature_importances_
#importances = 100.0 * (importances / importances.max())
#sorted_idx = np.argsort(importances)
#pos = np.arange(sorted_idx.shape[0]) + .5
#
#plt.barh(pos, importances[sorted_idx], align='center')
#plt.yticks(pos, train.drop(['Hazard'], axis=1).columns.values[sorted_idx])
#plt.xlabel('Relative Importance')
#plt.title('Variable Importance')
#plt.gcf().savefig('importances_%d.png' % code_id, format='png')
#plt.show()

# output predictions for submission
result = pd.DataFrame({'id': test.index})
result['Hazard'] = stacked.predict(test[columns])
result.to_csv('%d_stacked_result.csv' % code_id, index=False, sep=',')
