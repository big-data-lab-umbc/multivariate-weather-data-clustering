# -*- coding: utf-8 -*-
"""Sp_Agglo.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rffzeREHHxYtKe1WDhVz8nvwRHArX_Ob
"""

# !pip install "dask[dataframe]" Needed for data transformation

import pandas as pd
import numpy as np
import xarray as xr
import netCDF4 as nc
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering




  #Initialize parameters
  
K = int(input("Enter your desired number of clusters: "))

proximity = input("Enter your desired proximity metric(default is euclidean) : ")

link = input("Enter desired linkage criteria(from single, average, complete, ward), default is average: ")

components = int(input("Enter desired number of principal components: "))

replace = input("Would you like to replace values where there is land to null values\n")

def spatial_agglomerative(input):
  '''
  input: 
        datatype: 4-D spatio-temporal xarray

        n_clusters: The number of our desired clusters

        affinity: The distance or proximity metric used (euclidean in most cases)

        linkage: Linkage criteria

  Output:
         
        formed_clusters: 1-D array of cluster labels classifying each data point along the time dimension
                         to a cluster label

        A dataframe showing each cluster label and the correcponding cluster size.
     
  '''

  #calling function that inputs null values
 

  if replace == "yes":

    input = null_fill(input)

    #calling function that transforms our data
    trans_data = datatransformation(input)

    #Normalize data
    norm_data = datanormalization(trans_data)

    #calling function that reduces the dimension
    norm_data = pca1(norm_data, components)
  else:
    #calling function that transforms our data
    trans_data = datatransformation(input)

    #Normalize data
    norm_data = datanormalization(trans_data)

    #calling function that reduces the dimension
    norm_data = pca1(norm_data, components)
 

  # calling the agglomerative algorithm and choosing n_clusters = 4 based on elbow value
  model = AgglomerativeClustering(n_clusters = K, affinity = proximity, linkage = link)
  
  # training the model on transformed data
  y_model = model.fit(norm_data)
  labels = y_model.labels_
  
  # # creating pandas dataframe on transformed data
  # df2 = norm_data.time_step
  # df1 = pd.DataFrame(df2, columns=['index'])
  # df1['clusterid'] = labels
  
  # #df1["cluster"] = cluster.labels_
  # df1['clusterid'].value_counts()


  df1 = pd.DataFrame(norm_data)
  df1['Cluster'] = labels
  df1['Cluster'].value_counts()
  #print("Estimated number of clusters: %d" % n_clusters_)
  print(df1['Cluster'].value_counts())

  # graph size
  plt.figure(1, figsize = (24 ,12))

  # creating the dendrogram
  dendrogram = sch.dendrogram(sch.linkage(norm_data, method  = "ward"))

  plt.axhline(y = 85, color='orange', linestyle ="--")

  # var = list(input.variables)
  var = list(data.variables)

  # ploting graphabs
  plt.title('Dendrogram')
  plt.xlabel(var)
  plt.ylabel('Euclidean distances')
  plt.show()
  
  return df1,labels