#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 13:56:32 2021

@author: doan
"""
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
    #gl.xlabel_style = {'size':9, 'color':'gray'}
    #gl.ylabel_style = {'size':9, 'rotation':90, 'va':'center', 'color':'gray'}
    #plt.yticks(rotation=90)
    
    gl.ylabel_style = {'rotation': 0, 'size': 9, 'color': 'gray'}
    gl.xlabel_style = {'size': 9, 'color': 'gray'}
    gl.xpadding = 2


    return ax



                

if __name__ == '__main__':
    
    
    if False:
        var = 'PMSL'
        ifile = 'idata/PMSL_all_regr.nc'
    
    
        d0 = xr.open_dataset(ifile)[var] / 100.
        d1 = d0.loc['2005':'2010'] 
        d1.lon2d.values[d1.lon2d.values < 0] = d1.lon2d.values[d1.lon2d.values < 0] + 360
        #d1 = d1.isel(lat=slice(0,65),lon=slice(19,91)) 
        d1 = d1.isel(lat=slice(0,35),lon=slice(10,45)) 
        lats, lons = d1['lat2d'].values,d1['lon2d'].values 
    
    
        prmsl = d1[0].values
        
        
        #==============================
        # create Basemap instance.
        fig=plt.figure(figsize=(8,4.5))
        ax = fig.add_axes([0.05,0.05,0.9,0.85])
        
        m =Basemap(llcrnrlon=lons.min(),llcrnrlat=lats.min(),urcrnrlon=lons.max(),urcrnrlat=lats.max(),projection='mill')
        
        # contour levels
        clevs = np.arange(900,1100.,4.)
        x, y = m(lons, lats)
        cs = m.contour(x,y,prmsl,clevs,colors='k',linewidths=1.)
        
        m.drawcoastlines(linewidth=1.25)
        m.fillcontinents(color='0.8')
        m.drawparallels(np.arange(-80,81,20),labels=[1,1,0,0])
        m.drawmeridians(np.arange(0,360,10),labels=[0,0,0,1])
        
        plt.title('Mean Sea-Level Pressure (with Highs and Lows)' )
    
    
    
        plot_lows_highs(ax,m,x, y, prmsl)
        
        
        plt.show()
        
        






if __name__ == "__main__":
    print("Hello")
    
    
    
    
    
    
    
    
    
    
    
    