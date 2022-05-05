#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 18:54:03 2022

@author: doan
"""


from sklearn import cluster
from scipy.spatial import distance
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

def compute_bic(kmeans,X):
    """
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    BIC value
    """
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 
             'euclidean')**2) for i in range(m)])

    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return(BIC)



# IRIS DATA
iris = sklearn.datasets.load_iris()
X = iris.data[:, :4]  # extract only the features
#Xs = StandardScaler().fit_transform(X)
Y = iris.target


BIC = []
ks = range(1,10)
for i in ks:
    
    kmeans = cluster.KMeans(n_clusters = i, init="k-means++").fit(X) 
    # run 9 times kmeans and save each result in the KMeans object
    #KMeans = [cluster.KMeans(n_clusters = i, init="k-means++").fit(X) for i in ks]
    # now run for each cluster the BIC computation
    #bic_i = compute_bic(kmeans,X) 
    #def compute_bic(kmeans,X):
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, dim = X.shape


    #compute variance for all clusters beforehand
    cl_sums = []
    for j in range(m):
        xi = X[np.where(labels == j)]
        ci = centers[0][j]
        ed = ((xi - ci)**2).sum(axis=1)
        #dd = distance.cdist(xi, [ ci ], 'euclidean')**2
        cl_x = sum( ed ) 
        cl_sums.append(cl_x)
        
        
    
    cl_sum = sum(cl_sums) 
    cl_var = (1.0 / (N - m) / dim) * cl_sum
    

    const_term = 0.5 * m * np.log(N) * (dim+1)

    bic_ii = []
    for j in range(m):
        nbin = n[j]
        bic_j = nbin * np.log(nbin) - \
                nbin * np.log(N) - \
                ((nbin * dim) / 2) * np.log(2*np.pi*cl_var) - \
                ((nbin - 1) * dim/ 2)
                
        bic_ii.append(bic_j)
        
    bic_i = np.sum(bic_ii) - const_term
    
    
    #bic_i = np.sum([ n[i] * np.log(n[i]) -
    #                 n[i] * np.log(N) -
    #                 ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
    #                 ((n[i] - 1) * d/ 2) for i in range(m)]) - \
    #         const_term    
    
    
    BIC.append(bic_i)



print(BIC)
import matplotlib.pyplot as plt
plt.plot(BIC)








