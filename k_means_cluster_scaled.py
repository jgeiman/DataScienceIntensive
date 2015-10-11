#!/usr/bin/python 

""" 
    skeleton code for k-means clustering mini-project

"""




import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

from sklearn.preprocessing import MinMaxScaler

#def scale_features(features):
#    scaled = []    
#    for f in features:
#        f = Scale(f)
#        scaled.append(f)
#    return scaled

def scale_features(features):
    return Scale(np.array(features))
            
def Scale(arr):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(arr)    
    return scaler, scaled

def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than 4 clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()

### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)


### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
#feature_3 = "total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2]
#features_list = [poi, feature_1, feature_2, feature_3]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )

scaler, scaled_features = scale_features(finance_features)

print 'min: ', np.min(scaled_features, axis=0)
print 'max: ', np.max(scaled_features, axis=0)
print 'scale [$200K, $1M]:', scaler.transform(np.array([[2.e5,1.e6]]))

### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, line below assumes 2 features)
for f1, f2 in finance_features:
    plt.scatter( f1, f2 )
    
plt.show()



from sklearn.cluster import KMeans
features_list = ["poi", feature_1, feature_2]
#features_list = ["poi", feature_1, feature_2, feature_3]
data2 = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data2 )
scaler, scaled_features = scale_features(finance_features)

clf = KMeans(n_clusters=2)
#clf = KMeans(n_clusters=3)
pred = clf.fit_predict( scaled_features )
Draw(pred, scaled_features, poi, name="clusters_before_scaling.pdf", f1_name=feature_1, f2_name=feature_2)


### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred

try:
    Draw(pred, scaled_features, poi, mark_poi=False, name="scaled_clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "no predictions object named pred found, no clusters to plot"





