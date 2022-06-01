#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 16:19:11 2022

@author: doan
"""
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
import glob
import sklearn
import seaborn as sns
import string

from sklearn import metrics as mtr



outdir = '../output_20220318/'

mrt_func = {'ARS':mtr.adjusted_rand_score,
            'AMI':mtr.adjusted_mutual_info_score,
            'VMS':mtr.v_measure_score,
            'FMS':mtr.fowlkes_mallows_score,
            }


labs =  { 'ssim':'S-SIM', 'str':'COR', 'ed':'ED','md':'MD', 'rand':'Random'}
labs =  { 'ssim':'S k-means', 'str':'C k-means', 'ed':'E k-means','md':'M k-means', 'rand':'Random'}

ll = { 'S-SIM':'S k-means', 'COR':'C k-means', 'ED':'E k-means','MD':'M k-means', 'rand':'Random'}



sims = [ 'str', 'md',  'ssim', 'ed']
sims = [   'ssim', 'str','ed', 'md']
ini = 'rand'
inis = ['rand', 'pp']
kk = ['TC_ll']
kk = [ 'SLP_DJF',  'AM_t_1950_y', 'TC_ll'][:]
nrun = range(10)
sizes = range(2,21)


# write csv file
if 0:
    dd1 = {}
    dd2 = {}
    for ini in inis[:]:
        #===========================================
        for key in kk[:]:
            
            print(key)
            
            for size in sizes[:]:
                
                
                
                data = {}
                
                for sim in sims[:]: 
                    
                    od = outdir + '/'+key+'/'
                    
                    for ir in nrun: 
                        
                        odir = od +  '/%.2d'%ir +  '/n'+'%.2d' % size +'/'  + ini + '/'
                        ofile = odir +  sim+'.nc' 
                        do = xr.open_dataset(ofile)
                        data[(sim, ir)] = do.cluster.values

                dr = np.random.randint(size, size=do.cluster.values.size)
                
                
                #===========
                for sc in ['ARS', 'AMI', 'VMS', 'FMS'][:]:
                                    
                    measures = sims + ['rand']
                    print(size,sc)
                    for ir in nrun:
                        do1 = pd.DataFrame(index=labs.values(), columns = labs.values())
                        
                        for sim1 in measures[:]: 
                            for sim2 in measures[:]: 
                                if sim1 == 'rand':d0 = dr
                                else:d0 = data[(sim1,ir)]
                                if sim2 == 'rand': d0 = dr
                                else:d1 = data[(sim2,ir)]
                                
                                do1.loc[labs[sim1],labs[sim2] ] = mrt_func[sc](d0,d1)
                        
                        #=====      
                        dd1[(ini,key, sc, size,ir) ] = do1
                   
                        
                   #==============
                    for sim in sims:
                                    
                        nn = list(nrun) + [nrun[-1]+1]
                        nn_s = {n:'R'+str(n) for n in nn}
                        nn_s[nn[-1]] = 'Rand'
                        
                        do2 = pd.DataFrame(index=nn_s.values(), columns = nn_s.values() )
                        
                        for n1 in nn: 
                            for n2 in nn: 
                                if n1 == nn[-1]: d0 = dr
                                else:d0 = data[(sim,n1)]
                                if n2 == nn[-1]: d0 = dr
                                else:d1 = data[(sim,n2)]
                            
                                do2.loc[nn_s[n1],nn_s[n2] ] = mrt_func[sc](d0,d1)
                        
                        dd2[(ini,key, sc, size, sim) ] = do2
                                
                                
            
    dx1 = pd.concat(dd1)
    dx1.to_csv('fig/uncertainty_inter-measure.cvs')
    dx2 = pd.concat(dd2)
    dx2.to_csv('fig/uncertainty_inter-run.cvs')    










if 1:

    d1 = 1 - pd.read_csv('fig/uncertainty_inter-measure.cvs', index_col=[0,1,2,3, 4,5])
    d2 = 1 - pd.read_csv('fig/uncertainty_inter-run.cvs', index_col=[0,1,2,3, 4,5])

    # plot each piece ....
    for ini in inis[:0]:
        
        for sc in ['ARS', 'AMI', 'VMS', 'FMS'][:0]:   
            odir = 'fig_stat/' + ini+'_' + sc + '/'
            if not os.path.exists(odir): os.makedirs(odir) 
            
            
            for ik, key in enumerate(kk[:]):
                print(key)
            
                for size in sizes[:]:
                    
                    if 1:  
                        
                        for ir in nrun[:]:
                            #sns.set_theme()
                            d11 = d1.loc[ (ini, key, sc, size, ir) ]
                            df = d11.astype(float).abs().rename(index={'Random':'Rand'}).rename(columns={'Random':'Rand'})
                            #uniform_data = np.abs(df.astype(float).values) 
                            mask = np.zeros_like(df.values)
                            mask[np.triu_indices_from(mask)] = True
                            
                            
                            if 1:
                                
                                fig = plt.figure(figsize=[4,3])
                                ax = plt.axes([.15,.18,.8,.8])            
                                ax = sns.heatmap(df,annot=True, 
                                                 mask = mask,
                                                 vmin = 0.,
                                                 vmax=1.,
                                                 cbar_kws={'label': sc},
                                                 annot_kws={'size': 9},
                                                 fmt=".2f", center=.1)

                                ax.figure.axes[-1].yaxis.label.set_size(13)
                                plt.xticks( rotation=90, fontsize=9)
                                plt.yticks( rotation=0, fontsize=9)
                            
                                
                                ofile = odir + key + '_ims_k%.2d'%size+  '_r%.2d'%ir + '.png'
                                print(ofile)
                                fig.savefig(ofile, dpi = 150)
                                plt.close()
                        
                        
                        
                            
                    if 0:
                        for sim in sims:
                            print(sim)
                            d22 = d2.loc[(ini,key, sc, size, sim)]
                      
                            df = d22.astype(float).abs()
                            #uniform_data = np.abs(df.astype(float).values) 
                            mask = np.zeros_like(df.values)
                            mask[np.triu_indices_from(mask)] = True
                            
                            
                            if 1:
                                fig = plt.figure(figsize=[5,4])
                                ax = plt.axes([.1,.1,.8,.8])    
                         
                                ax = sns.heatmap(df,annot=True, 
                                             mask = mask,
                                             vmin = 0.,
                                             vmax=1.,
                                             cbar_kws={'label': sc},
                                             annot_kws={'size': 8},
                                             fmt=".1f", center=.1)   
                                
                                ax.figure.axes[-1].yaxis.label.set_size(15)
                                plt.xticks( rotation=90)
                                plt.yticks( rotation=0)
                        
                                ofile = odir + key + '_iru_k%.2d'%size+ '_' +sim+   '.png'
                                print(ofile)
                                fig.savefig(ofile, dpi = 150)    
                                plt.close()
    




    # plot combine 1 ...
    for ini in inis[:0]:
        for size in sizes[2:3]:
            
            
            for sc in ['ARS', 'AMI', 'VMS', 'FMS'][:]:   
                
                
                if 0:
                    for ir in nrun[:1]:
                        
                        odir = 'fig_stat/' 
                        if not os.path.exists(odir): os.makedirs(odir) 
                    
                        fig = plt.figure(figsize=[10,3.5])
                        exp = ['WP', 'CC', 'TC']    
                        for ik, key in enumerate(kk[:]):
                            
                            #sns.set_theme()
                            d11 = d1.loc[ (ini, key, sc, size, ir) ]
                            df = d11.astype(float).abs().rename(index={'Random':'Rand'}).rename(columns={'Random':'Rand'})
                            #uniform_data = np.abs(df.astype(float).values) 
                            mask = np.zeros_like(df.values)
                            mask[np.triu_indices_from(mask)] = True
                                
                            hs = .24
                            if ik == 2: hs = .3
                            ax = plt.axes([ik*.3 +.05,.2, hs,.7])   
                            cbar = False
                            if ik == 2: cbar = True
                            ax = sns.heatmap(df,
                                             annot=True, 
                                             mask = mask,
                                             vmin = 0.,
                                             vmax=1.,
                                             cbar_kws={'label': sc, "shrink": .5},
                                             annot_kws={'size': 9},
                                             fmt=".2f", center=.1, 
                                             cbar = cbar, 
                                             
                                             linewidths=.25, 
                                             linecolor='gray'
                                             )
                            
                            ax.axhline(y=0, color='gray',linewidth=.25)
                            ax.axhline(y=df.shape[1], color='gray',linewidth= 1)
                            ax.axvline(x=0, color='gray',linewidth=.25)
                            ax.axvline(x=df.shape[0], color='gray',linewidth=1)
    
    
                            ax.figure.axes[-1].yaxis.label.set_size(13)
                            plt.xticks( rotation=90, fontsize=9)
                            plt.yticks( rotation=0, fontsize=9)
                            
                            text =  string.ascii_lowercase[ik] 
                            ax.text(0.,1.01,text,fontsize=18, fontweight = 'bold', transform = ax.transAxes)
                            ax.text(1,1.01,exp[ik],fontsize=20, ha='right', transform = ax.transAxes)
                        
                            
                        ofile = odir + ini+'_compare_' + sc + '_ims_k%.2d'%size+  '_r%.2d'%ir + '.png'
                        print(ofile)
                        fig.savefig(ofile, dpi = 150)
                        
                            
                if 1:
                    
                    odir = 'fig_stat/' 
                    if not os.path.exists(odir): os.makedirs(odir)
                    for sim in sims[:]:
                        
                        fig = plt.figure(figsize=[10,3.5])
                        exp = ['WP', 'CC', 'TC']                            
                        for ik, key in enumerate(kk[:]):
                        
                            print(sim)
                            d22 = d2.loc[(ini,key, sc, size, sim)]
                      
                            df = d22.astype(float).abs()
                            #uniform_data = np.abs(df.astype(float).values) 
                            mask = np.zeros_like(df.values)
                            mask[np.triu_indices_from(mask)] = True
                    
                            
                         
                            hs = .24
                            if ik == 2: hs = .3
                            ax = plt.axes([ik*.3 +.05,.2, hs,.7])   
                            cbar = False
                            if ik == 2: cbar = True
                        
                    
  
                            ax = sns.heatmap(df,
                                             annot=True, 
                                             mask = mask,
                                             vmin = 0.,
                                             vmax=1.,
                                             cbar_kws={'label': sc, "shrink": .5},
                                             annot_kws={'size': 6},
                                             fmt=".1f", center=.1, 
                                             cbar = cbar, 
                                             linewidths=.25, 
                                             linecolor='gray'
                                             )       
                            
                            
                            ax.axhline(y=0, color='gray',linewidth=.25)
                            ax.axhline(y=df.shape[1], color='gray',linewidth= 1)
                            ax.axvline(x=0, color='gray',linewidth=.25)
                            ax.axvline(x=df.shape[0], color='gray',linewidth=1)
    
    
                            ax.figure.axes[-1].yaxis.label.set_size(13)
                            plt.xticks( rotation=90, fontsize=9)
                            plt.yticks( rotation=0, fontsize=9)
                            
                            text =  string.ascii_lowercase[ik] 
                            ax.text(0.,1.01,text,fontsize=18, fontweight = 'bold', transform = ax.transAxes)
                            ax.text(1,1.01,exp[ik],fontsize=20, ha='right', transform = ax.transAxes)
                        
                            
                        ofile = odir + ini+'_compare_' + sc + '_ir_k%.2d'%size+ '_'+  sim + '.png'
                        print(ofile)
                        fig.savefig(ofile, dpi = 150)



    
    


    # plot combine 2 ...
    for ini in inis[:0]:
        for size in sizes[2:3]:
            
            for ir in nrun[:1]:
                odir = 'fig_stat/' 
                fig = plt.figure(figsize=[9,6])
                for isc, sc in enumerate(['AMI', 'ARS']):   
                                    
                    exp = ['WP', 'CC', 'TC']    
                    for ik, key in enumerate(kk[:]):
                        
                        d11 = d1.loc[ (ini, key, sc, size, ir) ]
                        df = d11.astype(float).abs().rename(index={'Random':'Rand'}).rename(columns={'Random':'Rand'})
                        df = df.iloc[:4,:4]
                        mask = np.zeros_like(df.values)
                        mask[np.triu_indices_from(mask)] = True
                            
                        hs = .22
                        if ik == 2: hs = .28
                        ax = plt.axes([ik*.3 +.07,[.6,.1][isc], hs,.33])   
                        
                        
                        cbar = False
                        if ik == 2: cbar = True
                        ax = sns.heatmap(df,
                                         annot=True, 
                                         mask = mask,
                                         vmin = 0.,
                                         vmax=1.,
                                         cbar_kws={'label': sc, "shrink": .5},
                                         annot_kws={'size': 9},
                                         fmt=".2f", center=.1, 
                                         cbar = cbar,                    
                                         linewidths=.25, 
                                         linecolor='gray'
                                         )
                        
                        ax.axhline(y=0, color='gray',linewidth=.25)
                        ax.axhline(y=df.shape[1], color='gray',linewidth= 1)
                        ax.axvline(x=0, color='gray',linewidth=.25)
                        ax.axvline(x=df.shape[0], color='gray',linewidth=1)


                        ax.figure.axes[-1].yaxis.label.set_size(15)
                        plt.xticks( rotation=90, fontsize=9)
                        plt.yticks( rotation=0, fontsize=9)
                        
                        text =  string.ascii_lowercase[ik + 3*isc] 
                        ax.text(0.,1.02,text,fontsize=18, fontweight = 'bold', transform = ax.transAxes)
                        #ax.text(1,1.01,exp[ik],fontsize=20, ha='right', transform = ax.transAxes)
                        if isc ==0: 
                            ax.text(.5,1.04,exp[ik],
                                fontsize=20, ha='center', va = 'bottom', 
                                rotation = 0,
                                fontweight = 'bold',
                                bbox = dict(fc='lightgray', ec='none'),
                                transform = ax.transAxes)
                        
                ofile = odir + ini+'_compare_2sc_ims_k%.2d'%size+  '_r%.2d'%ir + '.png'
                print(ofile)
                fig.savefig(ofile, dpi = 150)
                        
                        
                        

    # plot combine 2 with chord (figure 8)
    
    for ini in inis[:0]:
        for size in sizes[::2]:
            
            for ir in nrun[:]:

                for isc, sc in enumerate(['AMI', 'ARS'][:1]):  
                    
                    
                    odir = 'fig/fig08/' 
                    if not os.path.exists(odir): os.makedirs(odir) 
                    fig = plt.figure(figsize=[9,6])                                    
                    exp = ['WP', 'CC', 'TC']    
                    for ik, key in enumerate(kk[:]):
                        
                        d11 = d1.loc[ (ini, key, sc, size, ir) ]
                        df = d11.astype(float).abs().rename(index={'Random':'Rand'}).rename(columns={'Random':'Rand'})
                        df = df.iloc[:4,:4]
                        cc = [ll[c].split()[0]   for c in df.index]
                        #cc[0] = 'S k-means'
                        df.columns = cc
                        df.index = cc
                        mask = np.zeros_like(df.values)
                        mask[np.triu_indices_from(mask)] = True
                            
                        hs = .22
                        if ik == 2: hs = .28
                        ax = plt.axes([ik*.3 +.07,.6, hs,.33])   
                        
                        
                        cbar = False
                        if ik == 2: cbar = True
                        ax = sns.heatmap(df,
                                         annot=True, 
                                         mask = mask,
                                         vmin = 0.,
                                         vmax=1.,
                                         cbar_kws={'label': 'CUD (-)', "shrink": .75},
                                         annot_kws={'size': 9},
                                         fmt=".2f", center=.1, 
                                         cbar = cbar,                    
                                         linewidths=.25, 
                                         linecolor='gray'
                                         )
                        
                        
                        ax.axhline(y=0, color='gray',linewidth=.25)
                        ax.axhline(y=df.shape[1], color='gray',linewidth= 1)
                        ax.axvline(x=0, color='gray',linewidth=.25)
                        ax.axvline(x=df.shape[0], color='gray',linewidth=1)


                        ax.figure.axes[-1].yaxis.label.set_size(15)
                        plt.xticks( rotation=90, fontsize=12)
                        plt.yticks( rotation=0, fontsize=12)
                        
                        text =  string.ascii_lowercase[ik + 3*isc] 
                        ax.text(0.,1.02,text,fontsize=18, fontweight = 'bold', transform = ax.transAxes)
                        #ax.text(1,1.01,exp[ik],fontsize=20, ha='right', transform = ax.transAxes)
                        if isc ==0: 
                            ax.text(.5,1.04,exp[ik],
                                fontsize=20, ha='center', va = 'bottom', 
                                rotation = 0,
                                fontweight = 'bold',
                                bbox = dict(fc='lightgray', ec='none'),
                                transform = ax.transAxes)
                        
                        
                        
                        val = df.values
                        for i in range(len(val)): val[i,i] = 0
                        
                        from plot_chord import chordDiagram
                        flux = val
                        #ax = plt.axes([0.05+.33*ik,0.1,.3,.9])
                        ax = plt.axes([ik*.3 +.07,.1, .22,.33])   
                        
                        #nodePos = chordDiagram(flux, ax, colors=[hex2rgb(x) for x in ['#666666', '#66ff66', '#ff6666', '#6666ff']])
                        nodePos = chordDiagram(flux, ax)
                        ax.axis('off')
                        prop = dict(fontsize=16, ha='center', va='center')
                        nodes = ['S-SIM', 'COR', 'ED', 'MD']
                        nodes = ['S k-means', 'C k-means', 'E k-means', 'M k-means']
                        nodes = ['S', 'C', 'E', 'M']
                        for i in range(4):
                            ax.text(nodePos[i][0], nodePos[i][1], nodes[i], rotation=nodePos[i][2], **prop)
                    
                        text =  string.ascii_lowercase[ik+3] 
                        ax.text(0.,1.,
                        #        exp[ik],
                                text,
                                fontsize=18, fontweight = 'bold', transform = ax.transAxes)
                        
                        #ax.text(.5,1.05,exp[ik],
                        #        fontsize=20, 
                        #        ha='center', transform = ax.transAxes)
                                            
                        
                        
                ofile = odir+ sc + '_k%.2d'%size +'r%.2d'%ir+  ini + '.png'
                        
                #ofile = odir + ini+'_compare_2sc_ims_k%.2d'%size+  'r%.2d'%ir + '.png'
                print(ofile)
                fig.savefig(ofile, dpi = 150)
                        
                        
                        
                                 
                        
                        
                        
    # plot combine 2 ...(figure 9)
    for ini in inis[:0]:
        
        for size in sizes[::2]:  

            for isc, sc in enumerate(['AMI', 'ARS'][:1]):   

                fig = plt.figure(figsize=[11,12])
                exp = ['WP', 'CC', 'TC']                           
                            
                odir = 'fig/fig09/' 
                if not os.path.exists(odir): os.makedirs(odir)  
                
                jj = [.725,.5,.275,.05]
                for isim, sim in enumerate(sims[:]):
                            
                    for ik, key in enumerate(kk[:]):
                            
                        print(sim)
                        d22 = d2.loc[(ini,key, sc, size, sim)].iloc[:-1,:-1]
                  
                        df = d22.astype(float).abs()
                        #uniform_data = np.abs(df.astype(float).values) 
                        mask = np.zeros_like(df.values)
                        mask[np.triu_indices_from(mask)] = True
                
                        
                        hs = .23
                        if ik == 2: hs = .28
                        ax = plt.axes([ik*.3 +.1,jj[isim], hs,.18])   
                        cbar = False
                        if ik == 2: cbar = True
                    
      
                        clab = 'CUD ('+ sc + ')'
                        
                        ax = sns.heatmap(df,
                                         annot=True, 
                                         mask = mask,
                                         vmin = 0.,
                                         vmax=1.,
                                         cbar_kws={'label': 'CUD (-)', "shrink": .75},
                                         annot_kws={'size': 6},
                                         fmt=".1f", center=.1, 
                                         cbar = cbar, 
                                         linewidths=.25, 
                                         linecolor='gray'
                                         )       
                        
                        
                        ax.axhline(y=0, color='gray',linewidth=.25)
                        ax.axhline(y=df.shape[1], color='gray',linewidth= 1)
                        ax.axvline(x=0, color='gray',linewidth=.25)
                        ax.axvline(x=df.shape[0], color='gray',linewidth=1)
    
    
                        ax.figure.axes[-1].yaxis.label.set_size(13)
                        plt.xticks( rotation=90, fontsize=9)
                        plt.yticks( rotation=0, fontsize=9)
                        
                        text =  string.ascii_lowercase[ik+3*isim] 
                        
                        ax.text(0.,1.01,text,fontsize=18, 
                                fontweight = 'bold', 
                                transform = ax.transAxes)
                        
                        if isim ==0: 
                            ax.text(.5,1.15,exp[ik],
                                fontsize=20, ha='center', 
                                rotation = 0,
                                fontweight = 'bold',
                                bbox = dict(fc='lightgray', ec='none'),
                                transform = ax.transAxes)
                    
                        if ik ==0: 
                            ax.text(-.2,.5,labs[sim],
                                fontsize=15, ha='right', va = 'center',
                                rotation = 90,
                                fontweight = 'bold',
                                bbox = dict(fc='lightgray', ec='none'),
                                transform = ax.transAxes)                        
                        

                ofile = odir+ sc + '_k%.2d'%size +  ini + '.png'
                
                print(ofile)
                fig.savefig(ofile, dpi = 150)



    
    
    
    
    
    
    # plot so sanh WP, CC, TC ... figure 10
    for ini in inis[:1]:
        for sc in ['ARS', 'AMI', 'VMS', 'FMS'][1:2]:   
            
            f1 = True
            if f1:
                fig = plt.figure(figsize=(10,6))
                ii, jj, hx, vx = [.1,.6,.3], [.6,.6,.1], .4, .33
                exp = ['WP', 'CC', 'TC']            
            
            for ik, key in enumerate(kk[:]):
                print(key)
            
                #===========================================
                
                    
                xx = {}
                vv = {}
                v2 = {}
                for size in sizes[:]:
                    
                    if 1:  
                        
                        for ir in nrun[:]:
                            #sns.set_theme()
                            d11 = d1.loc[ (ini, key, sc, size, ir) ]
                            df = d11.astype(float).abs()
                            #uniform_data = np.abs(df.astype(float).values) 
                            mask = np.zeros_like(df.values)
                            mask[np.triu_indices_from(mask)] = True
                            
                            
                            v = df.values[mask == 0]
                            vv[(size, ir)] = v[:-4]
                            
                            
                        
                        
                            
                    if 1:
                        for sim in sims:
                            print(sim)
                            d22 = d2.loc[(ini,key, sc, size, sim)]
                      
                            df = d22.astype(float).abs()
                            #uniform_data = np.abs(df.astype(float).values) 
                            mask = np.zeros_like(df.values)
                            mask[np.triu_indices_from(mask)] = True
                            
                            v = df.values[mask == 0]
                            v2[(size, sim)] = v[:-(df.shape[0]-1)]
                            
                            
                
                
                
                if 1:
                
                    # plotting
                    dx0 = pd.DataFrame(vv).mean()               
                    dx1 = pd.DataFrame(v2).mean()
                    
                    
                    #dx2 = dx1.mean()
                
                
                    if not f1: 
                        fig = plt.figure(figsize=(5,3))
                        ax = plt.axes([.1,.1,.8,.8])
                    else:
                        ax = plt.axes([ii[ik], jj[ik], hx, vx])
                        
                        #do.groupby(do.columns.get_level_values(0), axis=1).mean().mean().plot()
                        
                    cc = ['indianred', 'y',  'g', 'royalblue']
                    for isim, sim in enumerate(sims):
                        print(sim)
                        dp1 = dx1.loc[(slice(None), sim)]
                        ax.plot(dp1.index[::2], dp1[::2], color=cc[isim], lw=1, label='Inter-runs ('+labs[sim]+')')
                        
                        
                    for ir in nrun[:]:
                        dp0 = dx0.loc[(slice(None), ir)]
                        label = ''
                        if ir == 0: label= 'Inter-algorithms'
                        ax.plot(dp0.index[::2], dp0[::2], color='gray', lw=1, label=label)                    
                        
                        
                        
                        
                        
                        
                    plt.legend(frameon=False, ncol=2, fontsize=8)
                    xlabel = 'n-clusters'
                    
                    ax.set_ylabel( 'CUD ('+ sc+')', fontsize=15 )
                    ax.set_xlabel( xlabel, fontsize=13 )
                    #ax.text(0.,1.02,string.ascii_lowercase[ik]+') '+k,fontsize=18, fontweight = 'bold', transform = ax.transAxes)
                            
                    #ax.spines['right'].set_color('none')
                    #ax.spines['top'].set_color('none')
                    ax.xaxis.set_ticks_position('bottom')
                    
                    # was: axes.spines['bottom'].set_position(('data',1.1*X.min()))
                    ax.spines['bottom'].set_position(('axes', -0.0))
                    ax.yaxis.set_ticks_position('left')
                    ax.spines['left'].set_position(('axes', -0.0))
                    
                    ax.set_xlim([2, 20.5 ])
                    ax.set_ylim( [0,1] )
                    ax.xaxis.grid(False)
                    ax.yaxis.grid(False)
                    
                    
                    if f1: 
                        text =  string.ascii_lowercase[ik] 
                        ax.text(-0.1,1.1,text,fontsize=18, fontweight = 'bold', transform = ax.transAxes)
                        ax.text(.5,1.05,exp[ik],fontsize=20, ha='center', transform = ax.transAxes)
                    
                    
                    xt = np.arange(2,22,2) 
                    ax.set_xticks( xt )                    
                    
                    odir = 'fig/fig10/'
                    if not os.path.exists(odir): os.makedirs(odir)   
                    ofile = odir+'/u_'+sc+'.png'
                    fig.savefig(ofile, dpi = 150)  
                    
                    
                    
                    
                
            
            
            
            
            