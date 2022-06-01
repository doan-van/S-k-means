#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 12:38:53 2022

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




kk = [ 'SLP_DJF', 'AM_t_1950_y','TC_ll' ][:]
exp = { 'SLP_DJF':'WP', 
        'AM_t_1950_y':'CC', 
        'TC_ll': 'TC' }

sims = ['ssim', 'str', 'ed', 'md' ]
from scipy.stats import kurtosis, skew


label = {'ssim':'S-SIM', 'ed':'ED', 'str':'COR', 'md':'MD'}
color = {'ssim':'r', 'ed':'g', 'str':'y', 'md':'royalblue'}



iterables = [ [ exp[k] for k in kk], [label[sim] for sim in sims] ]
df = pd.DataFrame(columns=pd.MultiIndex.from_product(iterables, 
                                                     names=['first', 'second']), 
                  index=['MEAN', 'STD', 'SKEW', 'KUR', 'ENTROPY'])



def normalize1(x): return ( x - x.min() ) / ( x.max() - x.min())
def normalize2(x): return ( x ) / ( x.max() - x.min())




f1 = True
if f1:
    fig = plt.figure(figsize=(10,3))
    ii, jj, hh, vv = [.05,.35,.65], [.1,.1,.1], .25, .8
    
    
    
#==========================================
    
    
    
    
    
outdir = 'output_20220318'
    
    
    
#===========================================
for ik, key in enumerate(kk[:]):
    print(key)
    odir0 = '../' + outdir + '/' + key+'/' 


    if not f1: 
        fig = plt.figure(figsize=(4,4))
        ax = plt.axes([.1,.1,.8,.8])
    else:
        ax = plt.axes([ii[ik], jj[ik], hh, vv])    
        
        
    for sim in ['ssim', 'str', 'ed', 'md' ][:]: 
        ofile_sbtw = odir0 + sim+'_sim_btw.nc' 
        ds = xr.open_dataset(ofile_sbtw)
        
        x = ds.sim_btw.values
        
        
        xn = normalize1(x)
        
        
        
        x1 = xn[np.triu_indices(xn.shape[0],k=1)]
        print(sim)
        #ax.hist(x1,bins = np.arange(0.,1.,.01),
        #        label=label[sim],alpha=.4,color=color[sim], 
        #        histtype='stepfilled',lw=0.5,ec="k")
        #df.loc[:,(key,sim)] = x1.mean(), x1.std(), kurtosis(x1),skew(x1)
        import seaborn as sns
        sns.distplot(x1, hist=False, kde=True, ax=ax,norm_hist=False,
             bins=np.arange(0.,1.001,0.02), color = color[sim],
             hist_kws={'edgecolor':'none', 'alpha':.1},
             kde_kws={'linewidth': 2.5, 'alpha':.8}, label=label[sim])
        sns.distplot(x1, hist=True, kde=True, ax=ax,norm_hist=False,
             bins=np.arange(0.,1.001,0.02), color = color[sim],
             hist_kws={'edgecolor':'none', 'alpha':.1},
             kde_kws={'linewidth': 2.5, 'alpha':.8}, label='')

        x2 = np.histogram(x1, bins=   np.linspace(0,1,11))[0]
        x3 = x2/x2.sum()
        ent = -(x3*np.log2(x3)).sum()
        print(sim, ent)
        
        col = (exp[key], label[sim])
        df.loc['MEAN',col] = x1.mean()
        df.loc['STD',col] = x1.std()
        df.loc['KUR',col] = kurtosis(x1)
        df.loc['SKEW',col] = skew(x1)
        df.loc['ENTROPY',col] = ent
        
        
        
        
    plt.legend(loc=2, frameon=False)
    #ax.set_ylim(0,4e5)
    ax.set_ylabel('Probability Density')
    ax.set_xlabel('Normalized similarity')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('axes', -0.0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('axes', -0.0))
    ax.set_xlim([0, 1.01])
    ax.set_ylim([-0.0, 7])
    plt.yticks([])
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.ticklabel_format(style='plain')
    #ax.ticklabel_format(useOffset=False)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-4,4))
    #text =  string.ascii_lowercase[ik] 
    #ax.text(0.,1.01,text,fontsize=18, fontweight = 'bold', transform = ax.transAxes)
    #ax.text(.5,1.01,key,fontsize=13, ha='center', transform = ax.transAxes)
    #plt.grid(True)
    #fig.tight_layout()
    #plt.xticks(np.arange(0,101,20),['%.1f'%l for l in np.arange(0,1.01,.2)])        
        
        
    # was: axes.spines['bottom'].set_position(('data',1.1*X.min()))
    #ax.spines['bottom'].set_position(('axes', -0.05))
    #ax.yaxis.set_ticks_position('left')
    #ax.spines['left'].set_position(('axes', -0.05))
        
    if f1: 
        text =  string.ascii_lowercase[ik] 
        ax.text(-0.1,1.1,text,fontsize=18, fontweight = 'bold', transform = ax.transAxes)
        ax.text(.5,1.05,exp[key],fontsize=20, ha='center', transform = ax.transAxes)




odir = 'fig/fig02/'
if not os.path.exists(odir): os.makedirs(odir)   
fig.savefig(odir + 'S-distributions.png', dpi = 150)  
df.to_csv(odir + 'dability.csv', float_format='%.2f')
    

    
    
    
    
    
    
    