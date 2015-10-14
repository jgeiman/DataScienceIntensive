#!/usr/bin/python


"""
    starter code for the evaluation mini-project
    start by copying your trained/tested POI identifier from
    that you built in the validation mini-project

    the second step toward building your POI identifier!

    start by loading/formatting the data

"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 

from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import cross_validation

# train model
clf = tree.DecisionTreeClassifier()

# split the data into training and test data sets
features_train, features_test, labels_train, labels_test = \
    cross_validation.train_test_split(features, labels, test_size=0.3, 
                                      random_state=42)

# fit the decision tree to training data set
clf.fit(features_train, labels_train)

# predict labels
labels_pred = clf.predict(features_test)

# calculate accuracy
accuracy = accuracy_score(labels_test, labels_pred)
print 'Accuracy: ', accuracy
print 'Precision: ', precision_score(labels_test, labels_pred)
print 'Recall: ', recall_score(labels_test, labels_pred)
print 'F1: ', f1_score(labels_test, labels_pred)
