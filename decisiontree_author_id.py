#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 3 (decision tree) mini-project

    use an DT to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
from sklearn import tree
from sklearn.metrics import accuracy_score

# train model
clf = tree.DecisionTreeClassifier(min_samples_split=40)

print 'number of features: ', len(features_train[0])

t0 = time()
clf.fit(features_train, labels_train)
print "\ntraining time:", round(time()-t0, 3), "s"

# predict labels
t0 = time()
labels_pred = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"

# calculate accuracy
accuracy = accuracy_score(labels_test, labels_pred)
print 'Accuracy: ', accuracy

#########################################################

# OUTPUT

# Decision Tree with min_sample_split=40
# training time: 69.636 s
# prediction time: 0.018 s
# Accuracy:  0.97838452787
#########################################################

# number of features:  3785

#########################################################

# Using SelectPercentile = 1%
# number of features:  379
#
# training time: 4.719 s
# prediction time: 0.001 s
# Accuracy:  0.96700796359


