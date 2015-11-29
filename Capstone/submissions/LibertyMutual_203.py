import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from gini import Gini
from preprocess import encode_categorical

train = pd.read_csv('./data/train.csv', index_col='Id')
test = pd.read_csv('./data/test.csv', index_col='Id')

columns = train.drop(['Hazard'], axis=1).columns

# encode categorical variables as numbers
train = encode_categorical(train)
test = encode_categorical(test)

code_id = 203

print("Train a XGBoost model")
params = {"objective": "reg:linear",
          "n_estimators": 500,
          "learning_rate": 0.3,
          "max_depth": 5,
          "min_child_weight": 3,
          "silent": 1,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "seed": 47}

gbm = xgb.XGBRegressor(**params)

start_time = time.time()
gbm.fit(train[columns].as_matrix(), train.Hazard, eval_metric = 'auc')

stop_time = time.time()
print "training time: %.2fs" % (stop_time - start_time)

print("Make predictions:")
predictions = gbm.predict(test[columns].as_matrix())

haz_pred = gbm.predict(train[columns].as_matrix())
train_gini = Gini(haz_pred, train.Hazard)
print 'Gini (training):', train_gini

plt.scatter(train.Hazard, haz_pred)

## output predictions for submission
result = pd.DataFrame({'id': test.index})
result['Hazard'] = predictions
result.to_csv('%d_xgb_result.csv' % code_id, index=False, sep=',')
