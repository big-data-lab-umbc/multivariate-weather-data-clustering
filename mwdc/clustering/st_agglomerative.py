# -*- coding: utf-8 -*-
"""st_agglomerative.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rffzeREHHxYtKe1WDhVz8nvwRHArX_Ob
"""

import pandas as pd
import numpy as np
import xarray as xr
import netCDF4 as nc
from IPython.display import display
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn import cluster
from sklearn.metrics import *

from mwdc.preprocessing.preprocessing import data_preprocessing, pca1



def st_agglomerative(input_path, input, variables,n, K, affinity, linkage, p, transformation=True, dim_reduction=False, **kwargs):

  '''
  input parameters:

        input_path: path to your netCDF file

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

  data = xr.open_dataset(input_path, decode_times=False)
  var = list(data.variables)[3:]

  if transformation==True:

    #calling function that transforms our data
    norm_data = data_preprocessing(input_path, variables)

    if dim_reduction==True:
      #High dimension reduction
      norm_data = pca1(norm_data,n)

    else:
      
      if dim_reduction==False:
        print("")
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
      print(labels)
      print("")
      

      #var = list(data.variables)[3:]

      rmse = st_rmse(input_path, var, labels, transformation=True)
      print("This is the RMSE evaluation results:")
      print("")
      display(rmse)
      print("")
      spatial_corr = st_corr(input_path, var, labels, transformation=True)
      print("This is the Spatial Correlation evaluation results:")
      print("")
      display(spatial_corr)
      print("")
      silhouette_avg = silhouette_score(df1, labels)
      print("")
      print("For n_clusters =", K,"The average silhouette score is :", silhouette_avg)
      print("")

      davies_bouldin = davies_bouldin_score(df1, labels)
      print("")
      print("For n_clusters =", K,"The average davies bouldin score is :", davies_bouldin)
      print("")

      calinski_harabasz = calinski_harabasz_score(df1, labels) # It is also known as the Variance Ratio Criterion
      print("")
      print("For n_clusters =", K,"The average calinski harabasz score is :", calinski_harabasz) #Higher value of CH index means the clusters are dense and well separated, 
      #although there is no “acceptable” cut-off value.
      print("")
      print("")

      print(df1['Cluster'].value_counts())

    # graph size
      plt.figure(1, figsize = (18 ,12))
      
      # plot the top 7 levels of the dendrogram
      # No more than p levels of the dendrogram tree are displayed. A “level” includes all nodes with p merges from the last merge.
      plot_dendrogram(model, truncate_mode='level',p = 7, get_leaves=True, orientation='top', labels=None)

      plt.title('Hierarchical Clustering Dendrogram: ' + alg_name + ", " + metric)
      plt.xlabel('Sequence of Merges along the Time component - in Days')
      plt.ylabel(metric + " " +'distance')
      plt.show()

  return df1,labels



  #Example parameters to run code: With raw data and no PCA: st_agglomerative(path2,data, var, 7, 7, p=7, affinity="euclidean", linkage="average", transformation=True, dim_reduction=False)
  #                                With transformed data and PCA: st_agglomerative(trans_data,var, 7, 7, p=7, affinity="euclidean", linkage="average", transformation=True, dim_reduction=True)