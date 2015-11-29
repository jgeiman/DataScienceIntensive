# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 16:37:38 2015

@author: jag
"""
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone

class StackedClassiferRegressor(BaseEstimator, ClassifierMixin, RegressorMixin):
    
    def __init__(self, clf, reg=None, regs=[]):
        self.clf = clf
        self.base_reg = reg        
        self.regs = regs
        self._class_bins = None
        self._class_labels = None

    def fit(self, X, y, bins, labels=None):
        """
        Fit stacked classifer & regressor.

        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples,n_features]
            Training data
        
        y : numpy array of shape [n_samples, n_targets]
            Target values

        Returns
        -------
        self : returns an instance of self.
        """
        self._class_bins = bins
        self._class_labels = labels

        print "binning labels into classes"
        # create classes for y, based on provided bins and labels
        classes = pd.cut(y, bins, labels = labels)        
        self._classes = classes
        
        print "fit classifier to classes"
        # fit the classifier portion of stack        
        self.clf.fit(X, classes)
        
        for i, lbl in enumerate(labels):
            # use the portion of X with the given label            
            mask = (classes == lbl)
            
            if self.base_reg:
                # copy the base regressor
                r = clone(self.base_reg)
            else:
                r = self.regs[i]
            
            print "fitting regressor for class '%s'" % lbl
            # fit the regressor for this class
            r.fit(X[mask], y[mask])
 
            if self.base_reg:
               self.regs.append(r)
        
        return self
        
        
    def predict(self, X):
        """Predict regression target for X using stacked classifier and 
        regressor.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. 
            
        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted values.
        """
        # predict the class of y with classifier
        classes = self.clf.predict(X)
        
        # create default regressor predictions - zeros
        y_pred = np.zeros(X.shape[0])
        
        for lbl, r in zip(self._class_labels, self.regs):
            # use the portion of X with the given label            
            mask = (classes == lbl)
            
            if sum(mask) > 0:
                # fit the regressor for this class
                y_pred[np.array(mask)] = r.predict(X[mask])
        
        return y_pred    
    