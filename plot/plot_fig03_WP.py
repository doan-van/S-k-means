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
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import numpy as np
from sklearn.linear_model import LinearRegression
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker

sys.path.append('../')
from kmean import *
import os

def extrema(mat,mode='wrap',window=10):
    """find the indices of local extrema (min and max)
    in the input array."""
    mn = minimum_filter(mat, size=window, mode=mode)
    mx = maximum_filter(mat, size=window, mode=mode)
    # (mat == mx) true if pixel is equal to the local max
    # (mat == mn) true if pixel is equal to the local in
    # Return the indices of the maxima, minima
    return np.nonzero(mat == mn), np.nonzero(mat == mx)

def plot_lows_highs(ax,m,x, y,prmsl):
    # the window parameter controls the number of highs and lows detected.
    # (higher value, fewer highs and lows)
    local_min, local_max = extrema(prmsl, mode='wrap', window=50)


    xlows = x[local_min]; xhighs = x[local_max]
    ylows = y[local_min]; yhighs = y[local_max]
    lowvals = prmsl[local_min]; highvals = prmsl[local_max]
    
    # plot lows as blue L's, with min pressure value underneath.
    xyplotted = []
    # don't plot if there is already a L or H within dmin meters.
    yoffset = 0.022*(m.ymax-m.ymin)
    dmin = yoffset
    
    for x,y,p in zip(xlows, ylows, lowvals):
        if x < m.xmax and x > m.xmin and y < m.ymax and y > m.ymin:
            dist = [np.sqrt((x-x0)**2+(y-y0)**2) for x0,y0 in xyplotted]
            if not dist or min(dist) > dmin:
                ax.text(x,y,'L',fontsize=14,fontweight='bold',
                        ha='center',va='center',color='r')
                ax.text(x,y-yoffset,repr(int(p)),fontsize=9,
                        ha='center',va='top',color='r',
                        bbox = dict(boxstyle="square",ec='None',fc=(1,1,1,0.5)))
                xyplotted.append((x,y))
    # plot highs as red H's, with max pressure value underneath.
    xyplotted = []
    for x,y,p in zip(xhighs, yhighs, highvals):
        if x < m.xmax and x > m.xmin and y < m.ymax and y > m.ymin:
            dist = [np.sqrt((x-x0)**2+(y-y0)**2) for x0,y0 in xyplotted]
            if not dist or min(dist) > dmin:
                ax.text(x,y,'H',fontsize=14,fontweight='bold',
                        ha='center',va='center',color='b')
                ax.text(x,y-yoffset,repr(int(p)),fontsize=9,
                        ha='center',va='top',color='b',
                        bbox = dict(boxstyle="square",ec='None',fc=(1,1,1,0.5)))
                xyplotted.append((x,y))
       
                
       
        
       
        
       
#labs =  { 'ssim':'S-SIM', 'str':'COR', 'ed':'ED','md':'MD', 'rand':'Random'}          
labs =  { 'ssim':'S k-means', 'str':'C k-means', 'ed':'E k-means','md':'M k-means', 'rand':'Random'}
  



if __name__ == '__main__':

    
        
    # plot demonstration
    if 0:
        var = 'PMSL'
        ifile = '../idata/PMSL_all_regr.nc'
    
    
        d0 = xr.open_dataset(ifile)[var] / 100.
        d1 = d0.loc['2005':'2010'] 
        d1.lon2d.values[d1.lon2d.values < 0] = d1.lon2d.values[d1.lon2d.values < 0] + 360
        #d1 = d1.isel(lat=slice(0,65),lon=slice(19,91)) 
        d1 = d1.isel(lat=slice(0,35),lon=slice(10,45)) 
        lats, lons = d1['lat2d'].values,d1['lon2d'].values 
    
    
         
        
        for i in range(200)[:]:
                
            fig=plt.figure(figsize=(4,3))
            
            d2 = d1[i]
            
            lats, lons = d2['lat2d'].values,d2['lon2d'].values 
            z = d2.values
            proj =  ccrs.PlateCarree() 

            
            ax = plt.axes([.05, .05, .9,.9], projection= proj )
            ax.set_extent([115,165,20, 52])
            ax.coastlines(resolution='50m',lw=.0)
            
            #ax.text(1.0,1.02, 'C'+str(iz+1), 
            #        ha = 'right',
            #        fontsize=12, fontweight='bold', transform = ax.transAxes)
            
            #ax.gridlines()
            #ax.stock_img()
            #ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5,lw=.0)
            #ax.outline_patch.set_linewidth(.5)
            
            land_10m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m')
            ax.add_feature(land_10m, zorder=0, 
                           edgecolor='black', 
                           lw=1,
                           facecolor='wheat',alpha=.5)
            
            #ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.2,lw=.0)
            ax.outline_patch.set_linewidth(.2)
            
            clevs = np.arange(900,1100.,2.)
            cs = ax.contour(lons,lats,z,clevs,colors='gray', ls= '--',
                            linewidths=.75, transform= proj )                        
            plt.clabel(cs, clevs[::2], inline=1, fontsize=6, fmt='%.0f',)
            
            prmsl = d2.values
            # the window parameter controls the number of highs and lows detected.
            # (higher value, fewer highs and lows)
            local_min, local_max = extrema(prmsl, mode='wrap', window=50)                        
            
            xlows = lons[local_min]; xhighs = lons[local_max]
            ylows = lats[local_min]; yhighs = lats[local_max]
            lowvals = prmsl[local_min]; highvals = prmsl[local_max]                        
            
            
            # plot lows as blue L's, with min pressure value underneath.
            xyplotted = []
            # don't plot if there is already a L or H within dmin meters.
            yoffset = 0.022*(lats.max()-lats.min() )
            dmin = yoffset
            
            for x,y,p in zip(xlows, ylows, lowvals):
                print(x,y,p)
                if x <= lons.max() and x >= lons.min() and y <= lats.max() and y >= lats.min():
                    dist = [np.sqrt((x-x0)**2+(y-y0)**2) for x0,y0 in xyplotted]
                    print(dist)
                    if not dist or min(dist) > dmin:
                        print('plot')
                        ax.text(x,y,'L',fontsize=20,fontweight='bold',
                                ha='center',va='center',color='r',transform=ccrs.Geodetic())
                        #ax.text(x,y-yoffset,repr(int(p)),fontsize=9,
                        #        ha='center',va='top',color='r',
                        #        bbox = dict(boxstyle="square",ec='None',fc=(1,1,1,0.5)))
                        xyplotted.append((x,y))                        
            

            # plot highs as red H's, with max pressure value underneath.
            xyplotted = []
            for x,y,p in zip(xhighs, yhighs, highvals):
                if x <= lons.max() and x >= lons.min() and y <= lats.max() and y >= lats.min():
                    dist = [np.sqrt((x-x0)**2+(y-y0)**2) for x0,y0 in xyplotted]
                    if not dist or min(dist) > dmin:
                        ax.text(x,y,'H',fontsize=20,fontweight='bold',
                                ha='center',va='center',color='b')
                        #ax.text(x,y-yoffset,repr(int(p)),fontsize=9,
                        #        ha='center',va='top',color='b',
                        #        bbox = dict(boxstyle="square",ec='None',fc=(1,1,1,0.5)))
                        xyplotted.append((x,y))
                                        
        
        
            odir = 'fig/fig_wp/'
            ofile = odir + '%.3d'%i+'.png'
            plt.savefig(ofile, dpi=100)
            plt.close()
        

        
        
        

    for nr in range(10)[:1]:
        
        key = 'SLP_DJF'
        ini = 'rand'
        print(key)
        
        idir0 = '../output_20220318/'+key+'/'+  '%.2d'%nr +'/'
    
    
    
        fa = True
        if fa: 
            fig = plt.figure(figsize=(13, 13))
            ii0, jj0 = [.05,.55,.05,.55], [.53,.53,.04,.04]      
            
            odir = 'fig_20220318/'+key+'/'+ '%.2d'%nr +'/'
            if not os.path.exists(odir): os.makedirs(odir)               
            
        size = 4 
        for isim, sim in enumerate(['ssim', 'str', 'ed', 'md'][:]): 
            print(size)
            #n = size
            idir = idir0 +'/n'+'%.2d' % size +'/' + ini + '/'         
            ifile = idir + sim+'.nc' 
            

            ofile = odir +'/k-'+'%.2d' % size +'_'+sim+ '_' +ini+  '.png'    
        
            
            col = ['indianred', 'goldenrod', 'darkseagreen', 'steelblue']
            
            do = xr.open_dataset(ifile)
            lats, lons = do['lat2d'].values,do['lon2d'].values 
            zz = do.centroids
            
            if not fa:
                fig=plt.figure(figsize=(7,5.5))
                ii, jj, hh, vv = [.1,.6,.1,.6], [.6,.6,.1,.1], .4,.4
            else:
                
                ii, jj, hh, vv = np.array([0,.22, 0.,.22]) + ii0[isim], \
                                    np.array([.25, .25, 0.09,0.09]) + jj0[isim], .2,.2
            
            
            for iz, z in enumerate(zz[:]):
                
                # create Basemap instance.
                #fig=plt.figure(figsize=(5,5))
                
                proj =  ccrs.PlateCarree() 
                
                '''
                proj =  ccrs.LambertCylindrical(central_longitude=0)
                proj = ccrs.LambertConformal(central_longitude=lons.mean(), 
                                             central_latitude=lats.mean(), 
                                             false_easting=0.0, 
                                             false_northing=0.0, 
                                             secant_latitudes=None, 
                                             standard_parallels=None, 
                                             globe=None, cutoff=-30)
                '''
                
                ax = plt.axes([ii[iz], jj[iz], hh,vv], projection= proj )
                
                if iz ==0: 
                    ax.text(.0,1.02,labs[sim], 
                            fontsize=20, fontweight = 'bold', 
                            transform = ax.transAxes
                            )
                
                
                ax.set_extent([115,165,20, 52])
                ax.coastlines(resolution='50m',lw=.1)
                ax.text(1.0,1.02, 'C'+str(iz+1), 
                        ha = 'right',
                        fontsize=12, fontweight='bold', transform = ax.transAxes)
                #ax.gridlines()
                #ax.stock_img()
                #ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5,lw=.0)
                #ax.outline_patch.set_linewidth(.5)
                land_10m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m')
                ax.add_feature(land_10m, zorder=0, edgecolor='black', facecolor='wheat',alpha=.5)
                #ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.2,lw=.0)
                ax.outline_patch.set_linewidth(.2)
                
                
                clevs = np.arange(900,1100.,2.)
                cs = ax.contour(lons,lats,z,clevs,colors='gray', ls= '--',
                                linewidths=.75, transform= proj )                        
                plt.clabel(cs, clevs[::2], inline=1, fontsize=6, fmt='%.0f',)
                prmsl = z.values
                # the window parameter controls the number of highs and lows detected.
                # (higher value, fewer highs and lows)
                local_min, local_max = extrema(prmsl, mode='wrap', window=50)                        
                
                xlows = lons[local_min]; xhighs = lons[local_max]
                ylows = lats[local_min]; yhighs = lats[local_max]
                lowvals = prmsl[local_min]; highvals = prmsl[local_max]                        
                
                
                # plot lows as blue L's, with min pressure value underneath.
                xyplotted = []
                # don't plot if there is already a L or H within dmin meters.
                yoffset = 0.022*(lats.max()-lats.min() )
                dmin = yoffset
                
                for x,y,p in zip(xlows, ylows, lowvals):
                    print(x,y,p)
                    if x <= lons.max() and x >= lons.min() and y <= lats.max() and y >= lats.min():
                        dist = [np.sqrt((x-x0)**2+(y-y0)**2) for x0,y0 in xyplotted]
                        print(dist)
                        if not dist or min(dist) > dmin:
                            print('plot')
                            ax.text(x,y,'L',fontsize=14,fontweight='bold',
                                    ha='center',va='center',color='r',transform=ccrs.Geodetic())
                            #ax.text(x,y-yoffset,repr(int(p)),fontsize=9,
                            #        ha='center',va='top',color='r',
                            #        bbox = dict(boxstyle="square",ec='None',fc=(1,1,1,0.5)))
                            xyplotted.append((x,y))                        
                
    
                # plot highs as red H's, with max pressure value underneath.
                xyplotted = []
                for x,y,p in zip(xhighs, yhighs, highvals):
                    if x <= lons.max() and x >= lons.min() and y <= lats.max() and y >= lats.min():
                        dist = [np.sqrt((x-x0)**2+(y-y0)**2) for x0,y0 in xyplotted]
                        if not dist or min(dist) > dmin:
                            ax.text(x,y,'H',fontsize=14,fontweight='bold',
                                    ha='center',va='center',color='b')
                            #ax.text(x,y-yoffset,repr(int(p)),fontsize=9,
                            #        ha='center',va='top',color='b',
                            #        bbox = dict(boxstyle="square",ec='None',fc=(1,1,1,0.5)))
                            xyplotted.append((x,y))
                                
    
    
            #df = pd.read_csv(odir+sim+'_S-anal.csv', index_col=0)
            #fig = Plot_SS(df, colrs=col)
            df = pd.read_csv(idir+sim+'_S-anal.csv', index_col=0)
            if fa: ax = plt.axes([ii0[isim]+.1,jj0[isim],.15,.1])
            else: ax = plt.axes([.35,.1,.3,.25])
            Plot_SS(df, colrs=col, ax = ax)            
    
            
            if not fa: fig.savefig(ofile, dpi = 150)
            
        if fa: 
            
            odir = 'fig/fig03/'
            if not os.path.exists(odir): os.makedirs(odir)    
            
            ofile_a = odir + key + '%.2d'%nr  +'k%.2d' % size +'_' +ini+  '.png'    
            fig.savefig(ofile_a, dpi = 150)   
            #plt.close()



