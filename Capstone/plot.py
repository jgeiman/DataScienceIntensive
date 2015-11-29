# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 09:14:22 2015

@author: jag
"""

#import math

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import expon

from preprocess import encode_categorical

def hazard_jointplot(df, plot_dir='./plots/', threshold=None):
    
    if threshold:
        df = df[df.Hazard > threshold]
        print 'Generating plots for Hazard > ', threshold

    for c in train.columns.values:
        if c != 'Hazard':    
            print 'plotting : ', c
            g = sns.jointplot(x=c, y="Hazard", data=df, 
                              size=5, ratio=3, color="g")
            g.savefig('%s%s.png' % (plot_dir, c))            
    

def plot_exp_fit(series):
    # plot culumative mass function of the series
    fig = plot_cmf(series)

    # fit an exponential distribution and overlay the exponential fit
    loc, scale = expon.fit_loc_scale(series)
    dist = expon(loc=loc, scale=scale)
    x = np.linspace(0,70, 71)
    fig.gca().plot(dist.cdf(x), 'k--')
    
    return fig

def plot_cmf(data, data_label='', prob = None, units = ''):

    n = data.shape[0]
    cprob = (np.arange(n, dtype=np.float32) + 1)/(n + 1)
    sorted_data = np.sort(data)

    # Plot the data and curve fit    
    fig, ax = plt.subplots(1,1)
    ax.plot(sorted_data, cprob, c='0.75', alpha=0.75, linestyle='-', 
                linewidth=2.)
    ax.set_ylabel('Cumulative probability')
    ax.set_ylim((0, 1.))
    if data_label:
        ax.set_xlabel(data_label)
    return ax.figure
    

if __name__ == '__main__':
    train = pd.read_csv('./data/train.csv', index_col='Id')
    #test = pd.read_csv('./data/test.csv', index_col='Id')

    train = encode_categorical(train)

    #plot joint plots of each variable with Hazard
    hazard_jointplot(train)

    # Usage - to show only high hazard plots:
    #threshold = math.exp(3)
    #hazard_jointplot(train, './plots/high/', threshold=threshold)

    # plot a histogram of the hazard
    ax = sns.distplot(train.Hazard, fit=expon, kde=False)     
    ax.figure.savefig('./plots/hazard_hist.png')
    
    #plot the cumulative distibution & fit exponential curve
    g = plot_exp_fit(train.Hazard)
    g.savefig('./plots/hazard_exp_fit.png')
