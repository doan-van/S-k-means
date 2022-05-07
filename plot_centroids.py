#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 16:45:58 2021

@author: doan
"""

import glob, os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from numpy import random as rand
from kmean import *
from collections import OrderedDict
from sklearn import metrics
import sklearn


#===========================================
dp = pd.read_csv('idata/pattern_list/label_data.csv', index_col=0, parse_dates = True)






for key in  ['WPYS_all']:
    print(key)
    
    odir0 = 'output_20220201/'+key+'/' 
    #odir0 = 'output_20211218/'+key+'/' 
    dw = dp #[dp.index.month.isin( [12, 1, 2] )]

    
    scores = ['ARS', 'AMIS', 'VMS', 'FMS']
    iterables = [ scores, ['ssim', 'str',  'ed', 'md'] ]
    dd = pd.DataFrame(index = range(4,11), columns=pd.MultiIndex.from_product(iterables, names=['first', 'second']))

    for size in range(4,11)[:]:
        for sim in ['ssim', 'str', 'ed', 'md'][:]: 
            print(size)
            n = size
            odir = odir0 +'/n'+'%.2d' % n +'/'            
            ofile = odir + sim+'.nc' 
            do = xr.open_dataset(ofile)
            df = do.cluster.to_dataframe().astype(int)
            x = df.index.intersection(dw.index)
            d1 = df.loc[x].cluster.to_list()
            d0 = dw.loc[x,'gr'].to_list()
    
            #m = metrics.rand_score(d0, d1)
            a1 = sklearn.metrics.adjusted_rand_score(d0,d1)
            a2 = metrics.adjusted_mutual_info_score(d0,d1)
            a3 = metrics.v_measure_score(d0,d1)
            a4 = metrics.fowlkes_mallows_score(d0,d1)
            #a2 = sklearn.metrics.rand_score(d0,d1)
            dd.loc[size, ('ARS', sim) ] = a1
            dd.loc[size, ('AMIS', sim) ] = a2
            dd.loc[size, ('VMS', sim) ] = a3
            dd.loc[size, ('FMS', sim) ] = a4
            
            #print(sim,a1)
            #print(sim,a2)
            #print(sim,a3)
            #print(sim,a4)
            
            #plt.hist(d0,color='r', alpha=.3)
            #plt.hist(d1, alpha=.3)
            
    ylab = {'ARS': 'Adjusted Rand Score', 
            'AMIS': 'Adjusted Mutual Information',
            'VMS': 'V Measure', 
            'FMS': 'Fowlkes Mallows Score'} 
    for s in scores:
        print(s)
        fig = plt.figure(figsize=[6,3])
        d = dd.loc[:,s]
        
        ax = plt.axes([.1,.1,.8,.8])
        d.plot(ax=ax, kind='bar')
        ax.set_ylim(0,1)
        ax.legend(['SSIM', 'COR', 'ED', 'MD'])
        ax.set_xlabel('K (number of cluster)')
        ax.set_ylabel(ylab[s] )
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
            