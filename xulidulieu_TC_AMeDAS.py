#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 10:36:54 2022

@author: doan
"""


import pandas as pd
import glob, os, sys
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


if False:
    do = xr.Dataset()
    xx = []
    yy = []
    cc = []
    for y in range(1951,2021)[:]:
        print(y)    
        ifiles = sorted(glob.glob('idata/RMSC_data/itpl_besttrack/bwp'+str(y)+'/*.txt' ))
        for f in ifiles:
            print(f)
            df = pd.read_csv(f, header=None,delim_whitespace=True, index_col=0 )
            print(df.size)
            xx.append(df.values)
            cc.append(f.split('_')[-2])
            yy.append( int(f.split('_')[-1].split('.')[0]))
            
    do['TC'] = (('tc', 'p', 'll'), xx)
    do['year'] = ('tc', yy)
    do['code'] = ('tc', cc)
    do.to_netcdf('idata/RMSC_data/TC_combined.nc')
    




if True:
    ifile = '/Volumes/GoogleDrive/My Drive/share/working/K-means/scripts/idata/amedas_monthly.csv'
    df = pd.read_csv(ifile, index_col=[0,1,2])
     
    
    
    if False:
        xx = []
        for s in df.index.get_level_values(0).unique():
            
            x = df.loc[s].index.get_level_values(0).unique()
            x1 = pd.DataFrame(index = x )
            x1[s] = 1
            xx.append(x1)
            print(s, x[0])
            
        d = pd.concat(xx, axis=1)
        d.sum(axis=1).plot()
    
    
    am = pd.read_csv('/Users/doan/working/00_OBS_studies/AMEDAS/daily/Amedas_list.csv', index_col=0).set_index('station_id')
    
    
    for y in [1900, 1950, 1970][:1]: # np.arange(1900,1980,10)[:]:
        
        print(y)
        
        if False:
            ss = []
            for s in df.index.get_level_values(0).unique():
                x = df.loc[s].index.get_level_values(0).unique()
                if x[0] > y: continue
                print(s, x[0], y)
                ss.append(s)
                
            dr = xr.Dataset()
            do = df.loc[ss]
            vv = df.columns
            for v in vv[:1]: # ['temp_C', 'precip_mm']:
                d1 = do[v]
                xx = {}
                for s in d1.index.get_level_values(0).unique():
                    dd1 = d1.loc[s].loc[y:]
                    if dd1.isnull().sum() > 0: 
                        print(s, 'xxx')
                        #continue
                    if dd1.index.get_level_values(0).unique()[-1] != 2020: continue
                    xx[s] = dd1
                    
                d2 = pd.DataFrame(xx)
                d2 = d2.iloc[:,np.argwhere( (d2.isnull().sum() / len(d2) < 0.1).values )[:,0]]
                
                #d2.interpolate(axis = 0, inplace = True, limit_direction='both')
                
                d3 = d2.loc[d2.index.get_level_values(0).unique()[:30]]
                d4 = d3.groupby(d3.index.get_level_values(1)).mean()
                
                yy = { y:d2.loc[y] - d4  for y in d2.index.get_level_values(0).unique() }
                d5 = pd.concat(yy)
                d5.interpolate(axis = 0, inplace = True, limit_direction='both')
                dr[v] = ( ( 'sts', 'time'), d5.values.T )
            
            ame = am.loc[d2.columns]
            ame = ame[~ame.index.duplicated(keep='first')]
            
            times = pd.to_datetime(d2.index.get_level_values(0).astype('str')+'-'+d2.index.get_level_values(1).astype('str'))
            sts = d2.columns
            dr.coords['time'] = ('time', times)
            dr.coords['sts'] = ('sts', sts)
            for v1, v2 in zip( ['lat', 'lon', 'alt'], ['latitude', 'longitude', 'height' ]):
                dr[v1] = ('sts', ame.loc[:,v2])
            odir = 'idata/amedas/'
            if not os.path.exists(odir): os.makedirs(odir)  
            ofile = odir + 'from_'+str(y)+'.nc'
            dr.to_netcdf(ofile)
                
            
            
    if True:
        
        
        
        idir = 'idata/amedas/'
        ifile = idir + 'from_'+str(y)+'.nc'    
                  
        ds = xr.open_dataset(ifile).groupby('time.year').mean()
        ame = am.loc[ds.sts.values]
        ame = ame[~ame.index.duplicated(keep='first')]        
        
        
        
        plt.figure(figsize=(5, 2.5))
        ax = plt.axes( [.1,.1,.8,.8] ) 
            
            
        
        x = ds.year
        yy = ds.temp_C.values
        for y in yy.T:
            print(np.isnan(y).sum())
            ax.plot( x, y, color='gray', lw=.5)
        
        ax.plot(x,yy.mean(axis=1), color='indianred', lw=2)
        ds.close()
        
        
        ax.set_ylabel('$\Delta T (^oC)$')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
            
        # was: axes.spines['bottom'].set_position(('data',1.1*X.min()))
        ax.spines['bottom'].set_position(('axes', -0.05))
        ax.yaxis.set_ticks_position('left')
        ax.spines['left'].set_position(('axes', -0.05))
            
        ax.set_xlim([1900, 2020 ])
        ax.hlines(0, 1900, 2020, lw = 1, ls = '--')
        ax.set_ylim([-2,4])
        ax.xaxis.grid(False)
        ax.yaxis.grid(False)       
        
        
        
        
        
            
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy
        import numpy as np
        from sklearn.linear_model import LinearRegression
        from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
        import matplotlib.ticker as mticker
        
        def grid(ax,st):
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=2, color='gray', alpha=0.5, linestyle='--')
            gl.xlabels_top = False
            gl.ylabels_right = False
            gl.xlines = False
            gl.ylines = False
            gl.xlocator = mticker.FixedLocator(np.arange(0,360,st))
            gl.ylocator = mticker.FixedLocator(np.arange(-90.,90,st))
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style = {'size':9, 'color':'gray'}
            gl.ylabel_style = {'size':9, 'rotation':90, 'va':'center', 'color':'gray'}
            return ax
        
        
        #.sort_values('country_region_code')
        #proj = ccrs.PlateCarree(central_longitude=145)
        #proj._threshold /= 10
        proj =  ccrs.PlateCarree() #ccrs.Robinson(central_longitude=145)
        
        plt.figure(figsize=(4, 4))
        ax = plt.axes(projection= proj )
        ax.set_extent([127,150,26, 49])
        ax.coastlines(resolution='10m',lw=.5)
        #ax.gridlines()
        #ax.stock_img()
        
        ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5,lw=.0)
        ax.outline_patch.set_linewidth(.5)
        
        
        for i, row in ame.iterrows():
            ax.scatter(row.longitude,row.latitude,color='r',transform=ccrs.Geodetic(),s=10)    
            #x, y, fs, al = row.lon + row.x, row.lat + row.y, row.fs, row.al
            #text = row['text'].replace(';','\n')
            #plt.text( x, y, text, fontsize=fs, ha=al,
            #         transform=ccrs.Geodetic(), 
            #         )
        #grid(ax,5)
        #plt.savefig(out_filename, format='png', bbox_inches='tight', dpi = 200)
        
        
        ax.coastlines(resolution='10m',lw=.1, color='gray')
            
        land_10m = cartopy.feature.NaturalEarthFeature('physical', 'land', '10m')
        ax.add_feature(land_10m, zorder=0, edgecolor='black', facecolor='gray',alpha=.2)
        #ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.2,lw=.0)
            
        ax.outline_patch.set_linewidth(.0)












