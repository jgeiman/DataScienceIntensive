import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
#import xgboost as xgb

from gini import Gini
from preprocess import encode_categorical

train = pd.read_csv('./data/train.csv', index_col='Id')
test = pd.read_csv('./data/test.csv', index_col='Id')

columns = train.drop(['Hazard'], axis=1).columns

# encode categorical variables as numbers
train = encode_categorical(train)
test = encode_categorical(test)

# train a random forest
clf = RandomForestRegressor()
start_time = time.time()
clf.fit(train[columns], train['Hazard'])
stop_time = time.time()
print "training time: %.2fs" % (stop_time - start_time)

haz_pred = clf.predict(train[columns])
train_gini = Gini(haz_pred, train.Hazard)
print 'Gini (training):', train_gini
print 'Importances: ', clf.feature_importances_

# plot importances
importances = clf.feature_importances_
importances = 100.0 * (importances / importances.max())
sorted_idx = np.argsort(importances)
pos = np.arange(sorted_idx.shape[0]) + .5

plt.barh(pos, importances[sorted_idx], align='center')
plt.yticks(pos, train.drop(['Hazard'], axis=1).columns.values[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.gcf().savefig('importances_1.png', format='png')
plt.show()

## output predictions for submission
result = pd.DataFrame({'id': test.index})
result['Hazard'] = clf.predict(test[columns])
result.to_csv('1_RF_result.csv', index=False, sep=',')
