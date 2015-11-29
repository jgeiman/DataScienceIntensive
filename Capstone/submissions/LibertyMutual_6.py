import time

#import matplotlib.pyplot as plt
import pandas as pd
#import numpy as np
#from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

#from sklearn.cross_validation import StratifiedKFold
#from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.grid_search import GridSearchCV

from gini import Gini
from preprocess import encode_categorical
#from feature_selection import pca

train = pd.read_csv('./data/train.csv', index_col='Id')
test = pd.read_csv('./data/test.csv', index_col='Id')

columns = train.drop(['Hazard'], axis=1).columns

# encode categorical variables as numbers
train = encode_categorical(train)
test = encode_categorical(test)

code_id = 6
seed = 47

# train XGBoost regressor
#n = 500
#n_split = 75
#for n in (10, 50, 100):
print 'Running XGBoost...'
xgb_model = xgb.XGBRegressor()
clf = GridSearchCV(xgb_model,
                   {'max_depth': [2,4,6],
                    'n_estimators': [50,100,200]}, verbose=1)
                    
start_time = time.time()
clf.fit(train[columns], train['Hazard'])
stop_time = time.time()
print "training time: %.2fs" % (stop_time - start_time)

print 'Best score: ', clf.best_score_
print 'Best Parameters:', clf.best_params_

haz_pred = clf.predict(train[columns])
train_gini = Gini(haz_pred, train.Hazard)
print 'Gini (training):', train_gini
#print 'Importances: ', clf.feature_importances_

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

## output predictions for submission
result = pd.DataFrame({'id': test.index})
result['Hazard'] = clf.predict(test[columns])
result.to_csv('%d_XGB_result.csv' % code_id, index=False, sep=',')
