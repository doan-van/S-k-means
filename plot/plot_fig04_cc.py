"""
plot H's and L's on a sea-level pressure map
(uses scipy.ndimage.filters and netcdf4-python)
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from netCDF4 import Dataset
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import numpy as np
from sklearn.linear_model import LinearRegression
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import pandas as pd
from kmean import Plot_SS


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
    gl.xlabel_style = {'size':8, 'color':'gray'}
    gl.ylabel_style = {'size':8, 'rotation':90, 'va':'top', 'color':'gray'}
    return ax


    
labs =  { 'ssim':'S-SIM', 'str':'COR', 'ed':'ED','md':'MD', 'rand':'Random'}
labs =  { 'ssim':'S k-means', 'str':'C k-means', 'ed':'E k-means','md':'M k-means', 'rand':'Random'}

#am = pd.read_csv('/Users/doan/working/00_OBS_studies/AMEDAS/daily/Amedas_list.csv', index_col=0).set_index('station_id')
am = pd.read_csv('../../../amedas-down/Amedas_list.csv', index_col=0).set_index('station_id')

#===========================================
key = 'AM_t_1950_y'
ini = 'rand'

for nr in range(10)[:]:
    
    idir0 = '../output_20220318/'+key+'/'+  '%.2d'%nr +'/'
    
    di = xr.open_dataset('input_pp/'+key+'.nc' )
    x = di.idata.values
    shape = di.attrs['ishape']
    dims = di.attrs['idim']
    coords = di.coords
    yy = di.year.values
    
    idata = di.idata
    
    
    if 0:
        
        ame  = am.loc[di.sts] 
        
        
        proj =  ccrs.PlateCarree() #ccrs.Robinson(central_longitude=145)
        
        fig = plt.figure(figsize=(4, 6))
        #ax = plt.axes([.03,0.05,.7,.9], projection= proj )    
        #fig = plt.figure(figsize=(4, 4))

    
        ax = plt.axes( [.1,.22,.8,.25] ) 
        
        for ic, c in enumerate(idata):
            ax.plot(yy,c, color='lightblue', lw=1, alpha=1)
            #x1 = x[np.argwhere(clus == ic )[:,0]]
            #ax.fill_between(yy, x1.max(axis=0), x1.min(axis=0), alpha=.1, color=col[ic])
            #for x2 in x1:
            #    ax.plot(x2, color=col[ic], alpha=.1)
        ax.set_ylabel('$\mathrm{\Delta T\ (^oC)}$')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        
        # was: axes.spines['bottom'].set_position(('data',1.1*X.min()))
        ax.spines['bottom'].set_position(('axes', -0.02))
        ax.yaxis.set_ticks_position('left')
        ax.spines['left'].set_position(('axes', -0.02))
        ax.hlines(0, 1950, 2020, lw = 1, ls = '--', zorder=100)
        ax.set_xlim([1950, 2020 ])
        ax.set_ylim([-2,3])
        ax.xaxis.grid(False)
        ax.yaxis.grid(False)
                
        
        
        
        
        ax = plt.axes([.2,0.4,.6,.4], projection= proj )
            
        ax.set_extent([128,148,29,46])
        ax.coastlines(resolution='10m',lw=.1, color='gray')
        #ax.gridlines()
        #ax.stock_img()
        ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5,lw=.5)
        land_10m = cartopy.feature.NaturalEarthFeature('physical', 'land', '10m')
        
        ax.add_feature(land_10m, zorder=0, edgecolor='gray', 
                       lw=.5,
                       facecolor='none',alpha=.75)
        
        ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.2,lw=.0)
        ax.outline_patch.set_linewidth(.0)            
    
        for j, (i, row) in enumerate(ame.iterrows()) :
            ax.scatter(row.longitude,row.latitude,color='r',
                       edgecolor='k',lw=.1,
                       #transform=ccrs.Geodetic(),
                       s=10)            
        
        
    
    for size in range(4,21)[:1]:
        
        fa = True
        if fa: 
            fig = plt.figure(figsize=(10, 10))
            ii, jj = [.05,.55,.05,.55], [.55,.55,.05,.05]        
        
        for isim, sim in enumerate(['ssim', 'str', 'ed', 'md'][:]): 
            
            
            idir = idir0 +'/n'+'%.2d' % size +'/' + ini + '/'         
            ifile = idir + sim+'.nc' 
            

            
                
            if 1:
                
                do = xr.open_dataset(ifile)
                ame  = am.loc[do.sts] 
                
                
                proj =  ccrs.PlateCarree() #ccrs.Robinson(central_longitude=145)
                
                if fa:
                    #ax = plt.axes([.03,0.05,.7,.9], projection= proj )    
                    
                    ax = plt.axes([ii[isim], jj[isim], .43,.43], projection= proj )
                    ax.text(.02,0.93,labs[sim], 
                            fontsize=22, fontweight = 'bold', 
                            transform = ax.transAxes
                            )
                    
                else:
                    fig = plt.figure(figsize=(4, 4))
                    ax = plt.axes([.05,0.05,.9,.9], projection= proj )
                    
                ax.set_extent([126,150,25, 50])
                ax.coastlines(resolution='10m',lw=.1, color='gray')
                #ax.gridlines()
                #ax.stock_img()
                
                land_10m = cartopy.feature.NaturalEarthFeature('physical', 'land', '10m')
                ax.add_feature(land_10m, zorder=0, edgecolor='black', facecolor='gray',alpha=.2)
                #ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.2,lw=.0)
                
                ax.outline_patch.set_linewidth(.0)
    
                
                
                clus = do.cluster.values
                    
                col = ['indianred', 'goldenrod', 'darkseagreen', 'steelblue']
                
                cols = np.array(len(ame) * ['k'])
                for ic, co in enumerate(col): cols = np.where( clus == ic,  col[ic], cols)
                
                
                for j, (i, row) in enumerate(ame.iterrows()) :
                    ax.scatter(row.longitude,row.latitude,color=cols[j],
                               edgecolor='k',lw=.25,
                               #transform=ccrs.Geodetic(),
                               s=20)    
                    
                
                #grid(ax,5)
                if not fa:
                    ax = plt.axes( [.45,.1,.45,.2] ) 
                else:
                    ax = plt.axes([ii[isim]+.2, jj[isim] + .025, .2,.075 ])
                
                
                ax.hlines(0, 1950, 2020, lw = 1, ls = '--')
                for ic, c in enumerate(do.centroids[:]):
                    ax.plot(yy,c, color=col[ic], lw=1)
                    x1 = x[np.argwhere(clus == ic )[:,0]]
                    ax.fill_between(yy, x1.max(axis=0), x1.min(axis=0), alpha=.1, color=col[ic])
                    
                    #for x2 in x1:
                    #    ax.plot(x2, color=col[ic], alpha=.1)
                ax.set_ylabel('$\mathrm{\Delta T\ (^oC)}$')
                ax.spines['right'].set_color('none')
                ax.spines['top'].set_color('none')
                ax.xaxis.set_ticks_position('bottom')
                
                # was: axes.spines['bottom'].set_position(('data',1.1*X.min()))
                ax.spines['bottom'].set_position(('axes', -0.02))
                ax.yaxis.set_ticks_position('left')
                ax.spines['left'].set_position(('axes', -0.02))
                
                ax.set_xlim([1950, 2020 ])
                ax.set_ylim([-2,3])
                ax.xaxis.grid(False)
                ax.yaxis.grid(False)
                
        
                df = pd.read_csv(idir+sim+'_S-anal.csv', index_col=0)
                if fa:ax = plt.axes([ii[isim]+.3, jj[isim] + .125, .1,.1 ])
                else: ax = plt.axes([.65,.42,.22,.3])
                Plot_SS(df, colrs=col, ax = ax)            
        
                if not fa: 
                    odir = 'fig_20220318/'+key+'/'+ '%.2d'%nr +'/'
                    if not os.path.exists(odir): os.makedirs(odir)   
                    ofile = odir +'/k-'+'%.2d' % size +'_'+sim+ '_' +ini+  '.png'    
                    fig.savefig(ofile, dpi = 150)
            
        if fa: 
            
            odir = 'fig/fig04/'
            if not os.path.exists(odir): os.makedirs(odir)   
            ofile_a = odir +  key+ '_k'+'%.2d' % size +'r%.2d'%nr+'_' +ini+  '.png'    
            fig.savefig(ofile_a, dpi = 150)     








