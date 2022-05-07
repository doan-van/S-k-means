"""
plot H's and L's on a sea-level pressure map
(uses scipy.ndimage.filters and netcdf4-python)
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
#from mpl_toolkits.basemap import Basemap, addcyclic
from scipy.ndimage.filters import minimum_filter, maximum_filter
from netCDF4 import Dataset
import xarray as xr
import pandas as pd
from kmean import *
import glob 



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
    gl.ylabel_style = {'size':9, 'rotation':0, 'va':'center', 'color':'gray'}
    return ax



prob = {}
if True:
    ds = xr.open_dataset('idata/RMSC_data/TC_combined.nc')
    
    key = 'TC_ll'
    iput2d = ds.TC.values
    shape = iput2d.shape
    dims = ds.TC.dims
    iput1d = iput2d.reshape(shape[0],-1)
    prob[key] = {'idata': iput1d, 
                 'ishape': shape, 
                 'idim': dims,
                 'coords': ds.TC.coords}    
    
    


labs =  { 'ssim':'S-SIM', 'str':'COR', 'ed':'ED','md':'MD', 'rand':'Random'}
labs =  { 'ssim':'S k-means', 'str':'C k-means', 'ed':'E k-means','md':'M k-means', 'rand':'Random'}


#===========================================
key = 'TC_ll'
    
for nr in range(10):
    ini = 'rand'
    print(key)
    
    odir0 = 'output_20220201/'+key+'/' 
    
    di = xr.open_dataset('input_pp/'+key+'.nc' )
    
    
    xx = di.idata.values
    shape = di.attrs['ishape']
    dims = di.attrs['idim']
    coords = di.coords
    
    
    # plot demonstration
    if 0:
        # plot TC all
        proj =  ccrs.PlateCarree() #ccrs.Robinson(central_longitude=145)
                
        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes(projection= proj )
        ax.set_extent([110,180,5, 60])    
        
        land_10m = cartopy.feature.NaturalEarthFeature('physical', 'land', '10m')
        ax.add_feature(land_10m, zorder=0, edgecolor='black', facecolor='gray',alpha=.4, lw=0.3)
        ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5,lw=.5)
        ax.outline_patch.set_linewidth(.1)
        
        #y1 = xx[175:190,::2].mean(axis=0)
        #x1 = xx[175:190,1::2].mean(axis=0)

        #ax.plot(x1,y1, lw=2, c='k') 
        
        
        for xy in xx[:]:
            #print(xy)
            y = xy[::2]
            x = xy[1::2]
            ax.plot(x,y, lw=.5, c='red', alpha = .3) #row.latitude,color='r',transform=ccrs.Geodetic(),s=10)  
            #ax.scatter(x[0],y[0], c='g', s = 5)
    
    
        from functions import grid
        #grid(ax,10)
    
    
    
    for size in range(4,21)[:1]:
        
        idir0 = 'output_20220318/'+key+'/'+  '%.2d'%nr +'/'
        fa = 1
        if fa: 
            fig = plt.figure(figsize=(11, 8))
            ii, jj = [.05,.55,.05,.55], [.55,.55,.05,.05]
            
            
        for isim, sim in enumerate([ 'ssim', 'str', 'ed', 'md'][:]): 
            
            n = size
            
            #odir = odir0 +'/n'+'%.2d' % n +'/'  
            #odir = sorted(glob.glob('out/'+key+'/*'))[-1]+'/n'+'%.2d' % n +'/'  
            #ofile = odir +  sim+'.nc' 
            
            idir = idir0 +'/n'+'%.2d' % size +'/' + ini + '/'         
            ifile = idir + sim+'.nc' 
            
 
            
            
            do = xr.open_dataset(ifile)
            
            x = prob[key]['idata']
            
            col = ['indianred', 'goldenrod', 'darkseagreen', 'steelblue']
            #.sort_values('country_region_code')
            #proj = ccrs.PlateCarree(central_longitude=145)
            #proj._threshold /= 10
            proj =  ccrs.PlateCarree() #ccrs.Robinson(central_longitude=145)
            
            if fa:
                ax = plt.axes([ii[isim], jj[isim], .4,.4], projection= proj )
                ax.text(.01,0.93,labs[sim], 
                        fontsize=16, fontweight = 'bold', 
                        transform = ax.transAxes
                        )
                
                
            else:
                fig = plt.figure(figsize=(10, 5))
                ax = plt.axes(projection= proj )
            
            
            ax.set_extent([110,180,8, 50])
            #ax.coastlines(resolution='10m',lw=.5)
            #ax.gridlines()
            #ax.stock_img()
            land_10m = cartopy.feature.NaturalEarthFeature('physical', 'land', '10m')
            ax.add_feature(land_10m, zorder=0, edgecolor='black', lw=.5,
                           facecolor='gray',alpha=.4)
            #ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.2,lw=.0)
            ax.outline_patch.set_linewidth(.1)
            
            
            for xy in x[:0]:
                #print(xy)
                y = xy[::2]
                x = xy[1::2]
                if any(( (x > 126) & (x < 150)) &  ( (y > 25) & (y<45)) ):
                    ax.plot(x,y, lw=.1, c='gray') #row.latitude,color='r',transform=ccrs.Geodetic(),s=10)  
            
            ax.plot([126,150,150,126,126], [25,25,50,50,25], ls='--', c = 'k', lw=.5)
                
                
            clus = do.cluster.values
            
            for ic, cc in enumerate(do.centroids):
                y, x = cc[:,0], cc[:,1]
                ax.plot(x[:],y[:], lw=2, c=col[ic]) #row.latitude,color='r',transform=ccrs.Geodetic(),s=10)  
                ind = np.argwhere(clus == ic )[:,0] 
                x1 = xx[ind]
                for xy in x1[:]:
                    #print(xy)
                    y = xy[::2]
                    x = xy[1::2]      
                    ax.plot(x,y, lw=.1, c=col[ic], alpha=.5)

                
            grid(ax,10)
            #plt.savefig(out_filename, format='png', bbox_inches='tight', dpi = 200)
            

            df = pd.read_csv(idir+sim+'_S-anal.csv', index_col=0)
            if fa:ax = plt.axes([ii[isim]+.3, jj[isim] + .075, .12,.15 ])
            else:ax = plt.axes([.62,.25,.15,.3])
            Plot_SS(df, colrs=col, ax = ax)            
            
            #odir = 'fig/tc/'
            #if not os.path.exists(odir): os.makedirs(odir)   
            #ofile = odir +'/k-'+'%.2d' % size +'_'+sim+'.png'            
            if not fa: 
                odir = 'fig_20220318/'+key+'/'+ '%.2d'%nr +'/'
                if not os.path.exists(odir): os.makedirs(odir)   
                ofile = odir +'/k-'+'%.2d' % size +'_'+sim+ '_' +ini+  '.png'   
                fig.savefig(ofile, dpi = 150)  

        if fa: 
            
            odir = 'fig/fig05/'
            if not os.path.exists(odir): os.makedirs(odir)   
            ofile_a = odir +key + '_%.2d'%nr  + 'k'+'%.2d' % size +'_' +ini+  '.png'    
            fig.savefig(ofile_a, dpi = 150)              
            
        
            
        do.close()
    di.close()


