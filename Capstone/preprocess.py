# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 13:18:23 2015

@author: jag
"""
import pandas as pd
import numpy as np

def encode_categorical(df):
    """ 
    Encode categorical variables using the index of the sorted unique 
    values of the column.
    
    Example: ['N','Y'] gets mapped to [0, 1]
    """
    encoded = df.copy()
    for c in df.select_dtypes(include=['object']).columns.values:
        uniques = sorted(pd.unique(df[c]))
        col_map = {label: value for value, label in enumerate(uniques)}
#        print '\nEncoding %s using:\n%s' % (c, str(col_map))        
        encoded[c] = df[c].map(col_map)        
    return encoded

# code based on:
# http://www.ultravioletanalytics.com/2014/11/10/kaggle-titantic-competition-part-v-interaction-variables/
def drop_correlated(df, ignore_cols = ['Id','Hazard'], max_corr = 0.7):
    # calculate the correlation matrix
    df_corr = df.drop(ignore_cols, axis=1).corr(method='spearman')
     
    # create a mask to ignore self-
    mask = np.ones(df_corr.columns.size) - np.eye(df_corr.columns.size)
    df_corr = mask * df_corr
     
    drops = []
    # loop through each variable
    for col in df_corr.columns.values:
        # if we've already determined to drop the current variable, continue
        if np.in1d([col],drops):
            continue
        
        # find all the variables that are highly correlated with the current variable 
        # and add them to the drop list 
        corr = df_corr[abs(df_corr[col]) > max_corr].index
        drops = np.union1d(drops, corr)
     
    print "\nDropping", drops.shape[0], "highly correlated features...\n", drops
    return df.drop(drops, axis=1)
    
def encode_as_dummies(df):
    """ 
    Encode categorical variables using dummy variables for each category
    """
    cat_cols = df.select_dtypes(include=['object']).columns.values
    numeric = df.drop(cat_cols, axis=1)
    categories = pd.get_dummies(df[cat_cols])   
    return np.hstack((categories,numeric))
