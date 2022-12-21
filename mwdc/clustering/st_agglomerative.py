# -*- coding: utf-8 -*-
"""st_agglomerative.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qlHSso4uJkGT4D6t4WOIv_mbakEyl0I_
"""

import pandas as pd
import numpy as np
import xarray as xr
import netCDF4 as nc
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn import cluster

from mwdc.preprocessing.preprocessing import data_preprocessing, pca1

# from sklearn.metrics import adjusted_rand_score
# from sklearn.metrics import normalized_mutual_info_score
# from sklearn.metrics import *

def st_agglomerative(input_path, variables,n, K, affinity, linkage, p, transformation=True, **kwargs):

  '''
  input parameters:

        input_path1: path to your netCDF file

        input_path2: path to your netCDF file

        k : interger, The number of our desired clusters

        proximity: function, The distance metric

        linkage: function, Linkage criteria

        n: Integer Number of principal components

        p: int, (optional) The p parameter for truncate_mode.

        variables: List of netCDF variable you wish to use

        transformation: Boolean that accepts only "True" or "False"

  Output:
         
        formed_clusters: 1-D array of cluster labels classifying each data point along the time dimension
                         to a cluster label

        A dataframe showing each cluster label and the correcponding cluster size.

        A dendrogram showing the steps in clustering
     
  '''

  if transformation==True:

    #calling function that transforms our data
    norm_data = data_preprocessing(input_path, variables)

    #High dimension reduction
    norm_data = pca1(norm_data,n)

  else:
    
    if transformation==False:

      #High dimension reduction
      norm_data = pca1(input_path,n)


  def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
  
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,counts]).astype(float)
    
    # Plot the corresponding dendrogram
    
    dendrogram(linkage_matrix, **kwargs)

  #List of algorithms
  clustering_algorithms = (
      ('Single Linkage', 'single'),
      ('Average Linkage', 'average'),
      ('Complete Linkage', 'complete'),
      ('Ward Linkage', 'ward'))
  
  #distance metrics
  affinity_metrics = ['cosine', 'euclidean', 'manhattan']

  for metric in affinity_metrics:
    for alg_name, alg in clustering_algorithms:
      if alg == 'ward' and metric != 'euclidean': continue
      model = AgglomerativeClustering(n_clusters=K, affinity=metric, linkage=alg, compute_distances=True)
  
      #model.fit(data)
      y_model = model.fit(norm_data)
      labels = y_model.labels_

      df1 = pd.DataFrame(norm_data)
      df1['Cluster'] = labels
      df1['Cluster'].value_counts()


      #print("Estimated number of clusters: %d" % n_clusters_)
      print(df1['Cluster'].value_counts())

    # graph size
      plt.figure(1, figsize = (18 ,12))
      
      # plot the top four levels of the dendrogram
      #No more than p levels of the dendrogram tree are displayed. A “level” includes all nodes with p merges from the last merge.
      plot_dendrogram(model, truncate_mode='level',p = 7, get_leaves=True, orientation='top', labels=None)

      plt.title('Hierarchical Clustering Dendrogram: ' + alg_name + ", " + metric)
      plt.xlabel('Sequence of Merges along the Time component - in Days')
      plt.ylabel(metric + " " +'distance')
      plt.show()

  return df1,labels



  # var = list(data.variables)
  # var = var[3:]
  # var
  #Example parameters to run code: With raw data: st_agglomerative(path2,var, 7, 7, p=7, affinity="euclidean", linkage="average", transformation=True)
  #                                With transformed data: st_agglomerative(trans_data,var, 7, 7, p=7, affinity="euclidean", linkage="average", transformation=False)