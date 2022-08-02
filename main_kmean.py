#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 16:36:35 2022

@author: doan
"""

import glob, os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from numpy import random as rand
#from skimage.measure import compare_ssim as ssim
from kmean import *
from collections import OrderedDict

from datetime import datetime




outdir = 'output_20220318'
sys.exit()
import time
times = []


for nt in range(10)[:1]:
    
    date = pd.to_datetime(datetime.now()).strftime('%Y%m%d_%H%M%S')
    
    kk = ['TC_ll', 'AM_t_1950_y', 'SLP_DJF'][:]
    #===========================================
    for key in kk:
        print(key)
        
        
        odir0 = outdir + '/' + key + '/' + '%.2d' % (nt) +'/'
        if not os.path.exists(odir0): os.makedirs(odir0) 
        
        di = xr.open_dataset('input_data/'+key+'.nc' )
        
        x = di.idata.values
        shape = di.attrs['ishape'] 
        
        
        dims = di.attrs['idim']
        coords = di.coords
    
    
        for sim in ['ssim', 'ed', 'md', 'str'][:]: 
            # to calculate similarity in-between
            ofile_sbtw = outdir + '/' + key + '/'  + sim+'_sim_btw.nc' 
            
            if 1:
                
                ss = selsim(x, sim)
                do = xr.Dataset()
                do['sim_btw'] = (('from', 'to'), ss)
                
                try: do[dims[0]] = coords[dims[0]]
                except: print('coords')
                do.to_netcdf(ofile_sbtw)
                do.close()  
                    
                
            for size in range(2,21)[::2]:
                            
                
                n = size
                
                for ini in ['rand', 'pp'][:1]:
                    
                    odir = odir0 +'/n'+'%.2d' % n +'/'  + ini + '/'          
                    ofile = odir + sim+'.nc' 
                    
        
                    # run program for K means
                    # =======================
                    # ======
                    if 1:
        
                        if not os.path.exists(odir): os.makedirs(odir)  
                        
                        
                        start_time = time.time()
                        
                        o = K_means(x,k=n,sim=sim, ini=ini)
                        
                        duration =  time.time() - start_time
                        print(duration)
        
                        times.append( [nt,key,sim, size, ini,duration])
        
        
                        do = xr.Dataset(coords = coords ) 
                        C = o['C_fin']
                        s1 = [n] + list(shape[1:])
                        y = C.reshape( s1 )
                        do[ 'centroids' ] = (['n']+ list(dims[1:]),  y)
                        do['cluster'] = (dims[0],o['Clustering'])
                        
                        print(ofile)
                        do.to_netcdf(ofile)
                        do.close()  
        
        
                    if 1:
                        
                        print()
                        do = xr.open_dataset(ofile)
                        ds = xr.open_dataset(ofile_sbtw)
                        
                        ss = 1 - normalize1( ds['sim_btw'].values )
                        
                        clus = do['cluster']
                        df = Silh_s(x,ss,clus)
                        
                        df.to_csv(odir+sim+'_S-anal.csv')
                        print(odir+sim+'_S-anal.csv')
                        do.close()  
                        ds.close()
                        
                    if 1:
                        
                        df = pd.read_csv(odir+sim+'_S-anal.csv', index_col=0)
                        fig = Plot_SS(df)
                        
    
    
if 0: # write time record to file
    dt = pd.DataFrame(times, columns=['run', 'test', 'sim', 'size', 'ini', 'dur'])
    dt.to_csv(outdir + 'runtime.csv', index=None) 





