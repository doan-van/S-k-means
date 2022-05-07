#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Self-organizing map
# Doan Quang-Van
import glob, os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from numpy import random as rand
#from skimage.measure import compare_ssim as ssim
from kmean import *
from collections import OrderedDict



for f in ['PMSL_all.nc', 'UV500_DJF.nc'][:0]:
    ifile = 'idata/'+f
    ds = xr.open_dataset(ifile)
    ds.lon2d.values[ds.lon2d.values < 0] = ds.lon2d.values[ds.lon2d.values < 0] + 360
    dr = ds.rolling(lat=2).mean().rolling(lon=2).mean()
    do = dr.isel(lat=np.arange(1,dr.lat.size, 2), lon=np.arange(1,dr.lon.size, 2) )
    do.to_netcdf('idata/'+f+'_regr.nc')










var = 'PMSL'
ifile = 'idata/PMSL_all_regr.nc'
d0 = xr.open_dataset(ifile)[var] / 100.
d1 = d0.loc['2005':'2014'] 
#d1.lon2d.values[d1.lon2d.values < 0] = d1.lon2d.values[d1.lon2d.values < 0] + 360
#d1 = d1.isel(lat=slice(0,65),lon=slice(19,91)) 
d1 = d1.isel(lat=slice(0,35),lon=slice(10,45)) 
lat2d, lon2d = d1['lat2d'].values,d1['lon2d'].values 




prob = OrderedDict()


if 0:
    dp = pd.read_csv('idata/pattern_list/label_data.csv', index_col=0, parse_dates = True)
    key = 'YSN'
    d1 = d0.loc[dp.index]
    d1 = d1.isel(lat=slice(0,35),lon=slice(10,45)) 
    lat2d, lon2d = d1['lat2d'].values,d1['lon2d'].values 
    iput2d = d1.values 
    shape = iput2d.shape
    dims = d1.dims
    iput1d = iput2d.reshape(shape[0],-1)
    prob['SLP_'+key] = {'idata': iput1d, 
                            'ishape': shape, 
                            'idim': dims,
                            'coords': d1.coords}




if 0:
    #==========================================
    # Create input data
    d2 = {g[0]:g[1] for g in d1.groupby('time.season')}
    for key in ['DJF', 'JJA','MAM','SON']:
        d3 = d2[key]
        iput2d = d3.values 
        shape = iput2d.shape
        dims = d3.dims
        iput1d = iput2d.reshape(shape[0],-1)
        prob['SLP_'+key] = {'idata': iput1d, 
                            'ishape': shape, 
                            'idim': dims,
                            'coords': d3.coords}
      
        
        
    #======
    dp = pd.read_csv('idata/pattern_list/label_data.csv', index_col=0, parse_dates = True)
    dy = d0.sel(time=dp.index)
    dy2 = {g[0]:g[1] for g in dy.groupby('time.season')}
    dy2['all'] = dy
    
    for key in ['all' ,'DJF', 'JJA','MAM','SON']:
        d3 = dy2[key]
        iput2d = d3.values 
        shape = iput2d.shape
        dims = d3.dims
        iput1d = iput2d.reshape(shape[0],-1)
        prob['WPYS_'+key] = {'idata': iput1d, 
                            'ishape': shape, 
                            'idim': dims,
                            'coords': d3.coords}

#===========================================
#===========================================


if 0:
    for y in [1900, 1950, 1970][1:2]:
        f = 'idata/amedas/from_'+str(y)+'.nc'
        print(f)
        ds = xr.open_dataset(f)
        
        for iv in range(2)[:1]:
            v, v1 = ['temp_C', 'precip_mm'][iv], ['t', 'pr'][iv]
            dtyp = ['m', 'y']
            for ym in [0,1][:]:
                if ym: 
                    da = ds.groupby('time.year').mean().transpose()
                else: 
                    da = ds 
                key = 'AM_'+v1+'_'+str(y)+'_'+dtyp[ym]
                prob[key] = {'idata': da[v].values, 
                                'ishape': da[v].shape, 
                                'idim': da[v].dims,
                                'coords': da[v].coords}            






if 1:
    ds = xr.open_dataset('idata/RMSC_data/TC_combined.nc')
    
    key = 'TC_ll'
    iput2d0 = ds.TC.values
    xx = []
    for tc in iput2d0:
        y, x = tc[:,0], tc[:,1]
        if any(( (x > 126) & (x < 150)) &  ( (y > 25) & (y<45)) ):
            xx.append(tc)
    iput2d   = np.array(xx)[:,2:,:]
    shape = iput2d.shape
    dims = ds.TC.dims
    iput1d = iput2d.reshape(shape[0],-1)
    prob[key] = {'idata': iput1d, 
                 'ishape': shape, 
                 'idim': dims,
                 'coords': ds.TC.coords}    



#['SLP_DJF', 'SLP_JJA', 'SLP_MAM', 'SLP_SON', 
#'WPYS_all', 'WPYS_DJF', 'WPYS_JJA', 'WPYS_MAM', 'WPYS_SON']

# Define problem
# number of vector dimensions
#


for k, v in prob.items():
    print(k)
    odir = 'input_pp/' 
    ofile = odir + k + '.nc'
    do = xr.Dataset(coords = v['coords'] )
    do['idata'] = ( ('n_sample', 'n_dim'), v['idata'])
    do.attrs['ishape'] = v['ishape']
    do.attrs['idim'] = v['idim']
    do.to_netcdf(ofile)
    #do['ishape'] = ('ishape', np.array(v['ishape']) )
    #do['idim'] = ('idim', np.array(v['idim']) )
    

































