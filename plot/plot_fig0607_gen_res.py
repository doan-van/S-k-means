#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 16:20:58 2022

@author: doan
"""


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
import string
import matplotlib






kk = ['TC_ll', 'AM_t_1950_y', 'SLP_DJF'][:]
kk = [ 'SLP_DJF',  'AM_t_1950_y', 'TC_ll'][:]


    
    
    
labs = {'ssim':'S-SIM', 'str':'COR', 'ed':'ED', 'md': 'MD'} 
labs =  { 'ssim':'S k-means', 'str':'C k-means', 'ed':'E k-means','md':'M k-means', 'rand':'Random'}


kks = kk[:]
nrs = range(10)
sims = ['ssim', 'str', 'ed', 'md']
inis = ['pp', 'rand']
xn = range(2,21)


outdir = '../output_20220318/'
if 1:
    dd = {}
    #===========================================
    for ik, key in enumerate(kks):
        
        for nr in nrs:
            idir0 =  outdir + '/'+key+'/'+  '%.2d'%nr +'/'
            for sim in sims[:]: 
                for ini in inis:
                    aa = []
                    for size in xn[:]:     
                        idir = idir0 +'/n'+'%.2d' % size +'/' + ini + '/'         
                        df = pd.read_csv(idir+sim+'_S-anal.csv', index_col=0)
                        a = df.mean().values[0]
                        aa.append(a)
                    dd[(key,nr,sim,ini)] = aa
        
    do = pd.DataFrame(dd, index = xn)
    do.to_csv(outdir + 'SK_all.csv')    
    
    
    
    
    
    
    
    
df = pd.read_csv(outdir + '/SK_all.csv', header=[0,1,2,3], index_col=0 )
dt = pd.read_csv(outdir + '/runtime.csv', index_col=[0,1,2,3,4] )









for v in ['SK', 'RT'][:]:    
    f1 = True
    if f1:
        fig = plt.figure(figsize=(10,6))
        ii, jj, hh, vv = [.1,.6,.3], [.6,.6,.1], .4, .33
        exp = ['WP', 'CC', 'TC']
        
        
    
    ini = 'rand'
    for ik, key in enumerate(kks[:]):
        
        if v == 'SK':
            d1 = df.loc[:,(key, slice(None), slice(None), ini) ]
            do = d1.groupby(d1.columns.get_level_values(2), axis=1).mean()
            dos = d1.groupby(d1.columns.get_level_values(2), axis=1).std()
            ylim = [0,.7]
            ylabel = 'S-score'
            
        else:
            d1 = dt.loc[(slice(None), key, slice(None), slice(None),ini)  ]
            d2 = d1.reset_index().pivot(index="size", columns=['run', 'test', "sim", 'ini'], values="dur")
            
            for s in ['ssim', 'str', 'ed', 'md']:
                d3 = d2.loc[:,(slice(None), key, 'ed' )  ].values.copy()
                d2.loc[:,(slice(None), key, s) ] = d2.loc[:,(slice(None), key, s) ] / d3
            
            do = d2.groupby(d2.columns.get_level_values(2), axis=1).mean()
            dos = d2.groupby(d2.columns.get_level_values(2), axis=1).std()            
            ylim = [0,100]
            ylabel = 'Run time (sec.)'
        
        
        
        if not f1: 
            fig = plt.figure(figsize=(5,3))
            ax = plt.axes([.1,.1,.8,.8])
        else:
            ax = plt.axes([ii[ik], jj[ik], hh, vv])
            
        
        #do[['ssim', 'ed']].plot(ax=ax,kind='bar', color = ['r','g'],alpha=.5, edgecolor='k', width=0.5)
        #do[['ssim', 'str', 'ed', 'md']].plot(ax=ax, color = ['indianred', 'y',  'g', 'royalblue'],alpha=.75 , lw=1.5)
        cc = ['indianred', 'y',  'g', 'royalblue']
        for isim, sim in enumerate(['ssim', 'str', 'ed', 'md'][:] ):
            if v == 'RT' and sim == 'ed': continue
            dp = do[sim]
            x, y = dp.index, dp
            yerr = dos[sim]
            plt.errorbar(x[::2], y[::2], yerr=yerr[::2], 
                         c=cc[isim],lw=2, label = labs[sim] ) #, uplims=upperlimits, lolims=lowerlimits,
            #     label='subsets of uplims and lolims')
            
        
        if v == 'RT': ax.hlines(1,0,23, color = 'k', lw=1)
    
    
        plt.legend(frameon=False, ncol=2)
        xlabel = 'n-clusters'
        ax.set_ylabel( ylabel )
        ax.set_xlabel( xlabel )
        #ax.text(0.,1.02,string.ascii_lowercase[ik]+') '+k,fontsize=18, fontweight = 'bold', transform = ax.transAxes)
                
        #ax.spines['right'].set_color('none')
        #ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        
        # was: axes.spines['bottom'].set_position(('data',1.1*X.min()))
        ax.spines['bottom'].set_position(('axes', -0.0))
        ax.yaxis.set_ticks_position('left')
        ax.spines['left'].set_position(('axes', -0.0))
        
        ax.set_xlim([1.5, 20.5 ])
        if v != 'RT': ax.set_ylim([0,.6])
        ax.xaxis.grid(False)
        ax.yaxis.grid(False)
        
        
        if f1: 
            text =  string.ascii_lowercase[ik] 
            ax.text(-0.1,1.1,text,fontsize=18, fontweight = 'bold', transform = ax.transAxes)
            ax.text(.5,1.05,exp[ik],fontsize=20, ha='center', transform = ax.transAxes)
        
        if v == 'RT':
            ax.set_yscale('log')
            yt = [0.1,1, 5, 10]
            ax.set_yticks( yt )
            ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            #ax.get_yaxis().get_major_formatter().labelOnlyBase = False
        
        xt = np.arange(2,22,2) 
        ax.set_xticks( xt )


        odir = 'fig/fig06_07/'
        if not os.path.exists(odir): os.makedirs(odir)   
        ofile = odir + v + '.png'
        fig.savefig(ofile, dpi = 150)  

    
    











