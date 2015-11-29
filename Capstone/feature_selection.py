# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 16:06:39 2015

@author: jag
"""
from time import time

from sklearn.decomposition import RandomizedPCA

def pca(X_train, X_test, n):
    """Use PCA to perform unsupervised feature extraction."""

    print "Extracting %d principle components from %d features" % \
            (n, X_train.shape[1])
    t0 = time()
    pca = RandomizedPCA(n_components=n, whiten=True, random_state=47).fit(X_train)
    print "done in %0.3fs" % (time() - t0)
    
    print "Transforming the input data"
    t0 = time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print "done in %0.3fs" % (time() - t0)

    return X_train_pca, X_test_pca