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
from skimage.measure import compare_ssim as ssim
from kmean import *
from collections import OrderedDict
from sklearn import metrics
import sklearn




d1 = []
for i in range(10): 
    #[0,0,0,0,0,0,1,1,1]
    d1 = d1+[i]*20
  
#d1 = np.random.randint(0,10,200)
d0 = np.random.randint(0,10,200) # [1,1,1,2,2,2,1,1,1]

#m = metrics.rand_score(d0, d1)
a1 = sklearn.metrics.adjusted_rand_score(d0,d1)
a2 = metrics.adjusted_mutual_info_score(d1,d0)
a3 = metrics.v_measure_score(d0,d1)
a4 = metrics.fowlkes_mallows_score(d0,d1)
#a2 = sklearn.metrics.rand_score(d0,d1)
print('adjusted rand score ',a1)
print('multual information',a2)
print(a3)
print(a4)



            
            
            