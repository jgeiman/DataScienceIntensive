#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
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

#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 

#########################################################
### your code goes here ###
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# train model
#clf = SVC(kernel='linear')

#for c in (10, 100, 1000, 10000):    
c = 10000
print '\nrbf kernel, c =', c
clf = SVC(kernel='rbf', C=c)

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

# predict labels
t0 = time()
#labels_pred = clf.predict(features_test)
labels_pred = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"

#for i in [10,26,50]:
#    print "predicted label:", i, labels_pred[i]
#    print "actual labels:", i, labels_train[i]

print 'Chris emails: ', sum(labels_pred)
print 'Sara emails: ', 1700 - sum(labels_pred)

# calculate accuracy
accuracy = accuracy_score(labels_test, labels_pred)
print 'Accuracy: ', accuracy

#########################################################


# OUTPUT

## Linear kernel
#training time: 337.369 s
#prediction time: 13.999 s
#Accuracy:  0.98407281001

## Linear kernal, 1% of training data
#training time: 0.084 s
#prediction time: 0.828 s
#Accuracy:  0.88452787258

## RBF kernel
#training time: 0.085 s
#prediction time: 1.019 s
#Accuracy:  0.61604095563

#rbf kernel, c = 10
#training time: 0.089 s
#prediction time: 0.934 s
#Accuracy:  0.616040955631
#
#rbf kernel, c = 100
#training time: 0.084 s
#prediction time: 0.938 s
#Accuracy:  0.616040955631
#
#rbf kernel, c = 1000
#training time: 0.082 s
#prediction time: 0.886 s
#Accuracy:  0.82138794084
#
#rbf kernel, c = 10000
#training time: 0.079 s
#prediction time: 0.726 s
#Accuracy:  0.89249146757

## full training set
#rbf kernel, c = 10000
#training time: 93.797 s
#prediction time: 9.01 s
#Accuracy:  0.99089874857

## 1% training set, find predictions for 10, 26, 50
#rbf kernel, c = 10000
#training time: 0.084 s
#prediction time: 0.734 s
#predicted label: 10 1
#actual labels: 10 1
#predicted label: 26 0
#actual labels: 26 1
#predicted label: 50 1
#actual labels: 50 1
#Accuracy:  0.89249146757

# Chris emails vs Sara emails
#rbf kernel, c = 10000
#training time: 92.344 s
#prediction time: 8.935 s
#Chris emails:  877
#Sara emails:  823
#Accuracy:  0.99089874857