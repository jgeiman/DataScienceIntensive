#!/usr/bin/python


"""
    starter code for the validation mini-project
    the first step toward building your POI identifier!

    start by loading/formatting the data

    after that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

### it's all yours from here forward!  
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import cross_validation

# train model
clf = tree.DecisionTreeClassifier()

features_train, features_test, labels_train, labels_test = \
    cross_validation.train_test_split(features, labels, test_size=0.3, 
                                      random_state=42)

#t0 = time()
clf.fit(features_train, labels_train)
#print "\ntraining time:", round(time()-t0, 3), "s"

# predict labels
#t0 = time()
labels_pred = clf.predict(features_test)
#print "prediction time:", round(time()-t0, 3), "s"

# calculate accuracy
accuracy = accuracy_score(labels_test, labels_pred)
print 'Accuracy: ', accuracy
