#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 12:09:50 2021
Created on Sun May 10 02:33:27 2020
@author: doan
"""
import sys, os
from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



#==============================================================================
def strsim(x,y):
    term1 = 2*x.mean()*y.mean() / (x.mean()**2 + y.mean()**2)
    term2 = 2*np.cov(x.flatten(),y.flatten())[1,0] / (np.var(x)+np.var(y))
    return term1*term2
#==============================================================================


#==============================================================================
def strcnt(x,y):
    '''
    Parameters
    ----------
    x : input vector 1 (float)
        For comparison 
    y : input vector 2 (float)
        For comparison
    Returns
    -------
    Float
        Structural similarity index (-1 to 1)
    '''
    term1 = 1. #2*x.mean()*y.mean() / (x.mean()**2 + y.mean()**2)
    term2 = 2*np.cov(x.flatten(),y.flatten())[1,0] / (np.var(x)+np.var(y))
    return term1*term2
#==============================================================================
    


#==============================================================================
def S_luminance(x,y): return 2*x.mean()*y.mean() / (x.mean()**2 + y.mean()**2)
#==============================================================================    
#==============================================================================
def S_contrast(x,y): return 2* np.std(x) * np.std(y) / (np.var(x)+np.var(y))
#==============================================================================       
#==============================================================================
def S_structure(x,y): return np.cov(x.flatten(),y.flatten())[1,0] / (np.std(x) * np.std(y))
#==============================================================================
#==============================================================================
def edsim(x,y): return -np.linalg.norm(x - y)
#==============================================================================





sim_func = {'ssim':strsim,
            'ed': edsim,
            'lum':S_luminance,
            'cnt':S_contrast, 
            'str':S_structure, 
            'sc':strcnt}


#===================
def selsim(x, sim='ed'):
    # calculate self similarity in x vectors
    ss = np.zeros([x.shape[0],x.shape[0]])
    for i1, x1 in enumerate(x):
        #print(i1)
        for i2, x2 in enumerate(x): 
            if sim=='ssim': 
                ss[i2,i1] = strsim(x1, x2)
            if sim=='ed': 
                ss[i2,i1] = - np.linalg.norm(x1 - x2)
            if sim=='md': 
                ss[i2,i1] = - np.sum(np.abs(x1 - x2))                
            if sim == 'ssim_o': 
                from skimage.metrics import structural_similarity as ssim
                ss[i2,i1] = ssim(x1, x2,data_range = x2.max() - x2.min() )
            if sim in ['lum', 'cnt', 'str']:  ss[i2,i1] =  sim_func[sim](x1,x2 ) 
            
    return ss
#===================
    






#==============================================================================
#==============================================================================
def bmu1d(point,C,method='ssim'):
    '''
    * Find cluster that the point belongs to
    - point is a given point
    - C is cluster vectors
    - method is similarity method
    
    Find similarity (reverse of distance) from point to C vetors, 
    Cluster most similar (maximum value) to point will be returned
    '''
    
    #=======================
    # Structural simularity
    if method == 'ssim':
        
        values = []
        x = point
        for y in C[:]:
            term1 = 2*x.mean()*y.mean() / (x.mean()**2 + y.mean()**2)
            term2 = 2*np.cov(x.flatten(),y.flatten())[1,0] / (np.var(x)+np.var(y))
            values.append(term1*term2)
        values = np.array(values)
      
        
    #=======================
    # Structural similarity but using open source
    if method == 'ssim_o':
        x = point
        from skimage.metrics import structural_similarity as ssim
        # https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html
        values = np.array([ ssim(x, y, data_range = x.max() - x.min() )  for y in C[:]] )
        
        
    #=======================
    # If Eclidean distance, then the similairty with be nagative
    if method == 'ed':  
        #sub = C - point
        #values = - np.linalg.norm(sub, axis=1)
        x = point
        values = []
        for y in C[:]:
            values.append( - np.sqrt( np.nanmean((x-y)**2) ) ) # NOTE THAT NANMEAN INSTEAD OF SUM 
        values = np.array(values)          
    
    #========================
    # If Mahhatan distance, then the similairty with be nagative
    if method == 'md':  
        #sub = C - point
        #values = - np.linalg.norm(sub, axis=1)
        x = point
        values = []
        for y in C[:]:
            values.append( - np.sum(np.abs(x-y) ) )
        values = np.array(values)       
    
    
    #=======================
    # if luminance, contrast, structure (corelation)
    if method in ['lum', 'cnt', 'str']: 
        values = []
        x = point
        for y in C[:]:
            values.append( sim_func[method](x.flatten(),y.flatten()) )
        values = np.array(values)  
    
        
    #=======================
    maxv = np.max(values) 
    bmu_1d, smu_1d = np.argsort(values)[-1], np.argsort(values)[-2]
    
    return bmu_1d, smu_1d, maxv, values
#==============================================================================




def dist(a, b, ax=1): 
    #d = - np.sqrt( np.nanmean( (a-b)**2, axis = 1 ) ) 
    #return d
    return np.linalg.norm(a - b, axis=ax)
# function to compute euclidean distance 
def distance(p1, p2): return np.sum((p1 - p2)**2) 



# ====
# TO INITIALIZE CLUSTER
def initialize_rand(X,k):
    C = np.random.uniform(low=X.min(), high=X.max(), size=[k, X.shape[-1]] )
    return C
 
# ====
def initialize_pp(X,k):
    ''' 
    initialized the centroids for K-means++ 
    inputs: 
        data - numpy array of data points having shape (200, 2) 
        k - number of clusters  
    '''
    ## initialize the centroids list and add 
    ## a randomly selected data point to the list 
    centroids = [] 
    centroids.append(X[np.random.randint( X.shape[0]), :]) 
   
    ## compute remaining k - 1 centroids 
    for c_id in range(k - 1): 
          
        ## initialize a list to store distances of data 
        ## points from nearest centroid 
        dist = [] 
        for i in range(X.shape[0]): 
            point = X[i, :] 
            d = sys.maxsize 
              
            ## compute distance of 'point' from each of the previously 
            ## selected centroid and store the minimum distance 
            for j in range(len(centroids)): 
                temp_dist = distance(point, centroids[j]) 
                d = min(d, temp_dist) 
            dist.append(d) 
              
        ## select data point with maximum distance as our next centroid 
        dist = np.array(dist) 
        next_centroid = X[np.argmax(dist), :] 
        centroids.append(next_centroid) 
        dist = [] 

    return np.array(centroids)
#=====


def initialize_rand(X,k):
    ''' 
    initialized the centroids for K-means++ 
    inputs: 
        data - numpy array of data points having shape (200, 2) 
        k - number of clusters  
    '''
    # random samples
    c_ids = []
    n_sample = X.shape[0]
    if k >= n_sample:
        print('k is too large')
        sys.exit()
    for c_id in range(k): 
        
        cx = np.random.randint( n_sample)
        while cx in c_ids:
            print('one more')
            cx = np.random.randint( n_sample) 
        c_ids.append(cx)
                               
    
    C = X[c_ids]
    return C




ini_func = {'rand':initialize_rand,
            'pp': initialize_pp,
            }



#==============================================================================
def K_means(X,k=4,sim='ed', ini='rand'):
    X = np.array(X)
    
    if not ini in ['pp', 'rand']:
        
        print('\n Error: ini is not correct \n')
        sys.exit()
        
    C = ini_func[ini](X, k)
        
        
    print(C)
    
    # dict of output variables 
    ovar = {}
    ovar['C_ini'] = C
    
    
    
    # To store the value of centroids when it updates
    clusters = np.zeros(len(X))    
    cluster_all = [deepcopy(clusters)]
    C_all = [deepcopy(C)]
    
    for n in range(100):
        # Assigning each value to its closest cluster
        for i in range(len(X)): clusters[i],_,_,_ = bmu1d(X[i],C, method=sim)
        
        # if no vector is asigned to a cluster
        nvac =  np.in1d(np.arange(k), np.unique(clusters) )
        if not nvac.all():
            print(nvac)
            print('XXX')
        # still no solution
            
            
        #======
        
        # Storing the old centroid values
        C_old = deepcopy(C)
        # Finding the new centroids by taking the average value
        for i in range(k):
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            C[i] = np.nanmean(points, axis=0)
            if len(points) == 0: C[i] = C_old[i]
            
        #error = dist(C, C_old, None)
        error = np.sqrt( np.nanmean( (C-C_old)**2 ) ) 

        #print(n, error)
        
        cluster_all.append(deepcopy(clusters) )
        C_all.append(deepcopy(C))
        
        if np.isnan(error): 
            print('\n Error: error is nan \n')
            sys.exit()
            
        if error == 0.: break
    
        ovar['C_all'] = C_all
        ovar['Clustering_all'] = cluster_all    
    
    
    ovar['C_fin'] = C
    ovar['Clustering'] = clusters


    return ovar
#==============================================================================
    






#===================
# To evaluate 
def normalize1(x): return ( x - x.min() ) / ( x.max() - x.min())




# Calculate silholatte score
def Silh_s(x,ss,clus):
    #for sim in simi: ds[sim].values[:] = 1 - normalize1(ds[sim].values)
    soh = []
    for i in range(x.shape[0]):
        ithis = int(clus[i])
        sindex = ss[i]
                
        ab = np.array([sindex[clus == ig].mean() for ig in np.unique(clus) ])
        abis = np.argsort(ab) #[::-1]
        ai = ab[ithis]
        if abis[0] == ithis: bi = ab[abis[1]]    
        else: bi = ab[abis[0]]
    
        soi = (bi - ai) / max(bi, ai)
        if soi > 1.: print(soi, bi - ai, max(bi, ai))
        soh.append([ithis,soi])
        
    df = pd.DataFrame(soh)
    df.set_index(0,inplace=True) 
    return df
#================
    







#===============
def Plot_SS(df, ofile='fig.png', colrs = [], ax=None  ):
    
    if ax==None:
        fig = plt.figure(figsize=(3.8,3.8))
        ax = plt.axes([.1,.15,.84,.84])
    else:
        ax = ax
        
    y_lower, y_upper = 0, 0
    for ic in range(df.index.max()+1):
        try:
            cluster_silhouette_vals = df.loc[ic].values[:,0]
            cluster_silhouette_vals.sort()
            y_upper += len(cluster_silhouette_vals)
            if len(colrs) == 0:
                ax.barh(range(y_lower, y_upper), 
                    cluster_silhouette_vals, 
                    edgecolor='none', height=1, alpha=.8)
            else:
                ax.barh(range(y_lower, y_upper), 
                    cluster_silhouette_vals, color=colrs[ic],
                    edgecolor='none', height=1, alpha=.8)
                
            ax.text(-0.05, (y_lower + y_upper) / 2, str(ic + 1),va='center')
            y_lower += len(cluster_silhouette_vals)
        except:
            print('***')

    # Get the average silhouette score and plot it
    avg_score = df.mean().values[0]
    ax.axvline(avg_score, linestyle='--', linewidth=2, color='r')
    ax.set_yticks([])
    ax.set_xlim([-0.1, 1])
    #ax.set_xlabel('Silhouette coefficient values')
    ax.set_xlabel('S-score')
    ax.set_ylabel('Cluster labels')
    ax.spines['bottom'].set_position(('axes', -0.0))
    #ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('axes', -0.0))
    #ax.spines['right'].set_color('none')
    #ax.spines['top'].set_color('none')
    #ax.set_title('Silhouette plot for the various clusters', y=1.02);
    plt.setp(ax.spines.values(), linewidth=0.5)
    ax.xaxis.set_tick_params(width=0.5)
    # Scatter plot of data colored with labels
    #ofile = idir+'/fig_silhouette_'+sim+'.png'
    if ax==None: plt.savefig(ofile, dpi=150)
    #qper[sim].append( avg_score )
    return ax
#==============








if __name__ == "__main__":


    print('run demonstration problem')    
    odir0 = 'demo/'
    
        
    sim = 'ed'
    k = 4
    
    
    color = {'ed': 'k'}
    label = {'ed': 'ED'}
    
    
    nu_runs = 10
    
    for ir in range(nu_runs)[:]:
    
        odir  = odir0 + '/run_'+str(ir) + '/'
        if not os.path.exists(odir): os.makedirs(odir)
        
        
        x = np.random.normal( (.5,.5), .2, (300,2) )
        x = np.append(x, np.random.normal((1,1), .15, (200,2) ), axis=0)
        x = normalize1(x)
        dx = pd.DataFrame(x)
        input_file = odir + '/data_2d.csv'
        dx.to_csv(input_file)       

   
        if 1: # analyize and plot original data (S-distributions)
            
            if 1: # plot original data
                
                fig = plt.figure(figsize=(4,4))
                ax = plt.axes([.15,.15,.8,.8])
                    
                ax.scatter(x[:, 0], x[:, 1], s=15, c='gray', alpha = .8)
                ax.set_ylim(-.1,1.1)
                ax.set_xlim(-.1,1.1)
                ax.set_xlabel('Dim-1', fontsize=14)
                ax.set_ylabel('Dim-2', fontsize=14)
                xlab = np.arange(0,1.1,.2)
                xlabs = ['%.1f' % a for a in xlab]
                plt.yticks(xlab,xlabs, fontsize = 10)
                plt.xticks(xlab,xlabs, fontsize = 10)
                
                plt.savefig(odir+'/input_data.png', dpi=100)
                
                
            
            ss = - selsim(x, sim)
            xn = normalize1(-ss)
            x1 = xn[np.triu_indices(xn.shape[0],k=1)]
            
            
            
            import seaborn as sns
            
            if 1:
                
                print('plot S-distribution')
                def normalize1(x): return ( x - x.min() ) / ( x.max() - x.min())
                
                fig = plt.figure(figsize=[3,3])    
                
                ax = plt.axes([.15,.15, .8,.8])  
                
                
                sns.distplot(x1, hist=False, kde=True, ax=ax,norm_hist=False,
                     bins=np.arange(0.,1.001,0.02), color = 'k',
                     hist_kws={'edgecolor':'none', 'alpha':.2},
                     kde_kws={'linewidth': 2, 'alpha':.8}, label=label[sim])
                
                sns.distplot(x1, hist=True, kde=True, ax=ax,norm_hist=False,
                     bins=np.arange(0.,1.001,0.015), color = color[sim],
                     hist_kws={'edgecolor':'none', 'alpha':.1},
                     kde_kws={'linewidth': 0, 'alpha':.5}, label='')
        
                
                #plt.legend(loc=2, frameon=False)
                ax.set_ylabel('Probability distribution', fontsize=12)
                ax.set_xlabel('Normalized similarity', fontsize=12)
                ax.spines['right'].set_color('none')
                ax.spines['top'].set_color('none')
                ax.xaxis.set_ticks_position('bottom')
                ax.spines['bottom'].set_position(('axes', -0.0))
                ax.yaxis.set_ticks_position('left')
                ax.spines['left'].set_position(('axes', -0.0))
                ax.set_xlim([0, 1.01])
                ax.xaxis.grid(False)
                ax.yaxis.grid(False)
                ax.ticklabel_format(style='plain')
                #ax.ticklabel_format(useOffset=False)
                ax.ticklabel_format(axis='y', style='sci', scilimits=(-4,4))
                plt.xticks([0,.5,1])
                plt.yticks([],[])
                    
                plt.savefig(odir+'/S-D.png', dpi=100)
                
            
        
            
            
            # plot matrix
            if 1:
                fig = plt.figure(figsize=[5,4])
                                
                #odir = 'fig_stat/' 
                mask = np.zeros_like(xn)
                mask[np.triu_indices_from(mask)] = True
                    
                            
                         
                ax = plt.axes([.1,.1, .8,.8])   
                    
                ax = sns.heatmap(pd.DataFrame(xn[:100,:100]),
                                 #annot=True, 
                                 #mask = mask,
                                 #cmap="YlGnBu",
                                 vmin = 0.,
                                 vmax=1.,
                                 cbar_kws={'label': 'Normalized similarity', "shrink": .75},
                                 #annot_kws={'size': 6},
                                 #fmt=".1f", center=.1, 
                                 #cbar = cbar, 
                                 #linewidths=.25, 
                                 #linecolor='gray'
                                 )       
                
                
                ax.axhline(y=0, color='gray',linewidth=.25)
                plt.xticks( [],[])
                plt.yticks( [],[])
                
                plt.savefig(odir+'/S-matrix.png', dpi=100)
                   
        
    
                            
        if 1: # run k -means                        
            
        
            
            o = K_means(x,k=k,sim=sim, ini='rand')
            
            dcluster = pd.DataFrame({'Cluster':o['Clustering']})
            dcluster.to_csv(odir+'/cluster.csv')
            
         
            colors = ['tomato', 'lightgreen', 'royalblue', 'y', 'c', 'm']
            clusters_all = o['Clustering_all']
            
            for i, C in enumerate( o['C_all'][:] ):
                
                fig = plt.figure(figsize=(4,4))
                ax = plt.axes([.15,.15,.8,.8])
                #plt.scatter(x[:,0],x[:,1],alpha=.6,color='r',s=15) 
                
                for kk in np.unique(clusters_all[i]):
                    pp = np.array([x[j] for j in range(len(x)) if clusters_all[i][j] == kk])
                    ax.scatter(pp[:, 0], pp[:, 1], s=10, c=colors[int(kk)], alpha = .8)
                #plt.scatter(x[:,0],x[:,1],alpha=.6,color='r',s=15)  
                plt.scatter(C[:,0],C[:,1],marker='*',color='r',s=100,alpha=0.9) 
                ax.set_ylim(-.1,1.1)
                ax.set_xlim(-.1,1.1)
                ax.set_xlabel('Dim-1', fontsize=14)
                ax.set_ylabel('Dim-2', fontsize=14)
                xlab = np.arange(0,1.1,.2)
                xlabs = ['%.1f' % a for a in xlab]
                plt.yticks(xlab,xlabs, fontsize = 10)
                plt.xticks(xlab,xlabs, fontsize = 10)
       
                fig.savefig(odir+'%.2d'%(i+1)+'.png',dpi=100)
        
        
        
            clus = o['Clustering']
            
            ss = - selsim(x, sim)
            
            df = Silh_s(x,ss,clus)
            fig = plt.figure(figsize=(3,3))
            ax = plt.axes([.15,.15,.8,.8])
        
        
            Plot_SS(df, colrs=colors[:4], ax = ax)   
            fig.savefig(odir+'/Silhouete_plot.png',dpi=100)
            
        
            video = odir+'video.mp4'
            try:
                os.system('rm '+video)
                os.system('ffmpeg -r 2 -f image2 -s 1920x1080 -i '+odir+'/%02d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p '+video)
            except:
                print('fail to convert png to video')
        
        
    
    
    if True: # analize uncertainty
    
        import glob
        ifiles = sorted(glob.glob(odir0+'/run_*/cluster.csv'))
    
        xx = pd.DataFrame({ int(f.split('run_')[-1].split('/')[0]) : pd.read_csv(f, index_col=0).values[:,0] for f in ifiles })
        
        xx['rnd'] = np.random.randint(k, size=len(xx) )
        columns = xx.columns
        do = pd.DataFrame(index=columns, columns = columns)
        
        from sklearn import metrics as mtr
        
        for c1 in columns:
            for c2 in columns:
                print(c1, c2)
                d1, d2 = xx[c1].values, xx[c2].values
                do.loc[c1, c2] = 1 - mtr.adjusted_mutual_info_score(d1, d2)
                
        do.to_csv(odir0+'/CUD.csv')
        
        
        do1 = do.astype(float).abs()
        mask = np.zeros_like(do1.values)
        mask[np.triu_indices_from(mask)] = True
                    
                            
        fig = plt.figure(figsize=[5,4])
        ax = plt.axes([.15,.15,.8,.8])
        import seaborn as sns
                    
  
        ax = sns.heatmap(do1,
                         annot=True, 
                         mask = mask,
                         vmin = 0.,
                         vmax=1.,
                         cbar_kws={'label': 'CUD', "shrink": .5},
                         annot_kws={'size': 6},
                         fmt=".2f", center=.1, 
                         linewidths=.25, 
                         linecolor='gray'
                         )       
                        
        
        ax.axhline(y=0, color='gray',linewidth=.25)
        ax.axhline(y=do1.shape[1], color='gray',linewidth= 1)
        ax.axvline(x=0, color='gray',linewidth=.25)
        ax.axvline(x=do1.shape[0], color='gray',linewidth=1)


        ax.figure.axes[-1].yaxis.label.set_size(13)
        plt.xticks( rotation=90, fontsize=9)
        plt.yticks( rotation=0, fontsize=9)
        
        fig.savefig(odir0+'/CUD.png', dpi = 150)
                        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
    



