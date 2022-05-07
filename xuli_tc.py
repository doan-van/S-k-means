#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 13:58:07 2021

@author: doan
"""



#
# https://www.jma.go.jp/jma/jma-eng/jma-center/rsmc-hp-pub-eg/Besttracks/e_format_bst.html


import pandas as pd




if False:
    ll = open('idata/bst_all.txt').readlines()
    
    ind = [il for il, l in enumerate(ll[:]) if l[:5]=='66666']
    
    
    
    do = {}
    for i in range(len(ind))[:]:
        
        #+++++
        i1 = ind[i]
        if i == len(ind)-1: i2 = -1
        else: i2 = ind[i+1]
        #+++++
        
        t0 = ll[i1:i2]
        h = t0[0]
        bbbb, ccc, dddd, eeee, f, g, h20, i8 = h[6:10], h[12:15], h[16:20], h[21:25], h[26], h[28], h[30:50], h[64:72]
        #print(bbbb, ccc, dddd, eeee, f, g, h20, i8)
        print(h20)
        
        
        dd = []
        for l in t0[1:]:
            time, indicator, grade = l[:8], l[9:12], l[13:14]
            lat, lon, central_pres = l[15:18], l[19:23], l[24:28]
            
            max_ws, h, iiii = l[33:36], l[41], l[42:46]
            p = l[71]
            if int(time[:2]) > 30: time = '19'+time
            else: time = '20'+time
            dd.append([time, grade, lat, lon, central_pres])
            
        df = pd.DataFrame(dd, columns = ['time', 'grade', 'lat', 'lon', 'pres'])
        df.loc[:,'time'] = pd.to_datetime(df['time'], format='%Y%m%d%H')
        df.set_index('time', inplace=True)
        do[ (int(bbbb), h20.strip()) ] = df
    
    do = pd.concat(do)
    for c in ['grade', 'lat', 'lon', 'pres']: do.loc[:,c] = pd.to_numeric(do.loc[:,c], errors='coerce')
    
    do.loc[:,'lat'] = do.loc[:,'lat'] / 10.
    do.loc[:,'lon'] = do.loc[:,'lon'] / 10.
    
    
    do.to_csv('idata/tc_track_processed.csv' , float_format='%.2f')





#==========
if False:
    import numpy as np
    df = pd.read_csv('idata/tc_track_processed.csv', index_col=[0,1,2], parse_dates=[2])  
    
    x = [ '%.4d'% i for i in df.index.get_level_values(0) ]
    
    
    yy = np.array([ int(i[:2]) for i in x])
    df['y'] = np.where(yy>30, yy+1900, yy+2000)
    gr = list(df.groupby(df['y']))
         
    
    #gr = list(df.groupby(df.index.get_level_values(0)))
    odir = 'idata/tc_by_year/'
    import os
    if not os.path.exists(odir): os.makedirs(odir)
    
    for g in gr:
        print(g[0])
        g[1].to_csv(odir+str(g[0])+'.csv' )
        


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import glob, sys, os  
    
fig   =  plt.figure(figsize=(6,6))    
ax  = plt.axes([0.1,0.2,0.8,0.75], projection=ccrs.PlateCarree()) #central_longitude=180))
ax.set_extent( [80,180,-20,60] )
#ax.set_extent((-120, 120, -45, 45))
arg = {'color':'k', 'lw':.5}
ax.coastlines(resolution = '10m',**arg)
    
    
ifiles = sorted(glob.glob('idata/tc_by_year/*.csv'))

for f in ifiles[-10:]:
    df = pd.read_csv(f, index_col=[0,1,2], parse_dates=[2])  
    gr = df.groupby(df.index.get_level_values(0))
    
    for g in gr:
        print(g[0])
        d = g[1]
        
        ax.plot(d['lon'], d['lat'], lw=1)

    
    
    
