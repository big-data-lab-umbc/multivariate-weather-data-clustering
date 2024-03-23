# -*- coding: utf-8 -*-
"""Parea_multi_view_ensemble_clustering.ipynb


Paper: Parea: multi-view ensemble clustering for cancer subtype discovery

https://paperswithcode.com/paper/parea-multi-view-ensemble-clustering-for


Code: https://github.com/mdbloice/Pyrea/tree/master
"""



from google.colab import drive
drive.mount('/content/drive')

"""#Install needed libraries"""

# Install dask.dataframe
!pip install "dask[dataframe]"
!pip install netCDF4

!git clone https://""@github.com/

# Commented out IPython magic to ensure Python compatibility.
# %cd multivariate-weather-data-clustering

!python setup.py install

import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
#import netCDF4
# from netCDF4 import Dataset
# import netCDF4 as nc
import time
import random
from scipy import linalg as LA
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans, KMeans, SpectralClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import seaborn as sns
import xarray as xr
#from mwdc.clustering.st_agglomerative import st_agglomerative

import warnings
warnings.filterwarnings("ignore")


from mwdc.visualization.clusterplotting import clusterPlot2D
from mwdc.visualization.visualization import visualization2
from mwdc.preprocessing.preprocessing import data_preprocessing
from mwdc.evaluation.st_evaluation import st_rmse_df, st_corr, st_rmse_np
from mwdc.clustering.st_agglomerative import st_agglomerative
from yellowbrick.cluster import KElbowVisualizer

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

import sys

import matplotlib as mpl
import matplotlib.colors as colors
import os
import xarray as xr
import warnings
warnings.filterwarnings("ignore")

!pip install pyrea

! pip install netCDF4

import netCDF4 as nc
import pandas as pd
import numpy as np
import xarray as xr
import datetime
import datetime as dt
from netCDF4 import date2num,num2date
from math import sqrt

from sklearn.metrics import silhouette_samples, silhouette_score
def silhouette_score1(X, labels, *, metric="cosine", sample_size=None, random_state=None, **kwds):
 return np.mean(silhouette_samples(X, labels, metric="cosine", **kwds))

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

def data_preprocessing(data_path):
  rdata_daily = xr.open_dataset(data_path)    # data_path = '/content/drive/MyDrive/ERA5_Dataset.nc'
  rdata_daily_np_array = np.array(rdata_daily.to_array())   # the shape of the dailt data is (7, 365, 41, 41)
  rdata_daily_np_array_latitude = np.concatenate((rdata_daily_np_array, np.zeros((7, 365, 41,7), dtype=int)), axis=3)
  rdata_daily_np_array_longitude = np.concatenate((rdata_daily_np_array_latitude, np.zeros((7, 365, 7, 48), dtype=int)), axis=2)
  rdata_daily_np_array = rdata_daily_np_array_longitude
  rdata_daily_np_array_T = rdata_daily_np_array.transpose(1,0,2,3)   # transform the dailt data from (7, 365, 41, 41) to (365, 7, 41, 41)
  overall_mean = np.nanmean(rdata_daily_np_array_T[:, :, :, :])
  for i in range(rdata_daily_np_array_T.shape[0]):
    for j in range(rdata_daily_np_array_T.shape[1]):
      for k in range(rdata_daily_np_array_T.shape[2]):
        for l in range(rdata_daily_np_array_T.shape[3]):
          if np.isnan(rdata_daily_np_array_T[i, j, k, l]):
            #print("NAN data in ", i, j, k, l)
            rdata_daily_np_array_T[i, j, k, l] = overall_mean
  rdata_daily_np_array_T = rdata_daily_np_array_T.transpose(0,2,3,1)
  rdata_daily_np_array_T_R = rdata_daily_np_array_T.reshape((rdata_daily_np_array_T.shape[0], -1))  # transform the dailt data from (365, 7, 41, 41) to (365, 11767)
  min_max_scaler = preprocessing.MinMaxScaler() # calling the function
  rdata_daily_np_array_T_R_nor = min_max_scaler.fit_transform(rdata_daily_np_array_T_R)   # now normalize the data, otherwise the loss will be very big
  #rdata_daily_np_array_T_R_nor = np.float32(rdata_daily_np_array_T_R_nor)    # convert the data type to float32, otherwise the loass will be out-of-limit
  rdata_daily_np_array_T_R_nor_R = rdata_daily_np_array_T_R_nor.reshape((rdata_daily_np_array_T_R_nor.shape[0], 1, rdata_daily_np_array.shape[2], rdata_daily_np_array.shape[3], rdata_daily_np_array.shape[0]))
  return rdata_daily_np_array_T_R_nor, rdata_daily_np_array_T_R_nor_R

import pyrea
import numpy as np

def parea(data):
  # Create two datasets with random values of 1000 samples and 100 features per sample.
  d1 = data
  d2 = data

  # Define the clustering algorithm(s) you want to use. In this case we used the same
  # algorithm for both views. By default n_clusters=2.
  c1 = pyrea.clusterer('hierarchical', n_clusters=7, method='ward')
  c2 = pyrea.clusterer('hierarchical', n_clusters=7, method='complete')
  c3 = pyrea.clusterer('hierarchical', n_clusters=7, method='single')
  c4 = pyrea.clusterer('spectral', n_clusters=7, random_state=10, n_init=30)
  c5 = pyrea.clusterer('spectral', n_clusters=7, random_state=10, n_init=30)
  c6 = pyrea.clusterer('spectral', n_clusters=7, random_state=10, n_init=30)


  # Create the views using the data and the same clusterer

  v1 = pyrea.view(d1, c1)
  v2 = pyrea.view(d1, c2)
  v3 = pyrea.view(d1, c3)
  v4 = pyrea.view(d1, c4)
  v5 = pyrea.view(d1, c5)
  v6 = pyrea.view(d1, c6)
  v7 = pyrea.view(d2, c1)
  v8 = pyrea.view(d2, c2)
  v9 = pyrea.view(d2, c3)
  v10 = pyrea.view(d2, c4)
  v11 = pyrea.view(d2, c5)
  v12 = pyrea.view(d2, c6)

  # Create a fusion object
  f = pyrea.fuser('disagreement')

  # Specify a clustering algorithm (precomputed = True)
  c_pre = pyrea.clusterer("hierarchical", n_clusters=7, method='complete', precomputed=True)
  # Execute an ensemble based on your views and a fusion algorithm
  v_res = pyrea.view(pyrea.execute_ensemble([v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12], f), c_pre)

  # The cluster solution can be obtained as follows
  labels = v_res.execute()
  labels = labels.reshape(labels.shape[0])
  return labels

"""# **Evaluation Metrics**

**Davies bouldin**
"""

def davies_bouldin_score(X, labels):
 return print("Davies-Bouldin score is ", davies_bouldin_score(X, labels))

"""**Calinski Harabasz**"""

def calinski_harabasz_score(X, labels):
 return print("Calinski Harabasz score is ", calinski_harabasz_score(X, labels))

"""**RMSE**"""

def total_rmse(data_path,formed_clusters):
  #processed_data = data_preprocessing(data_path)
  processed_data = data_nor
  trans_data = pd.DataFrame(processed_data)
  trans_data['Cluster'] = formed_clusters

  # Normalized
  # Function that creates two dictionaries that hold all the clusters and cluster centers
  def nor_get_clusters_and_centers(input,formed_clusters):
    Clusters = {}
    Cluster_Centers = {}
    for i in set(formed_clusters):
      Clusters['Cluster' + str(i)] = np.array(input[input.Cluster == i].drop(columns=['Cluster']))
      Cluster_Centers['Cluster_Center' + str(i)] = np.mean(Clusters['Cluster' + str(i)],axis=0)
    return Clusters,Cluster_Centers

  intra_rmse = []
  sq_diff = []
  Clusters,Cluster_Centers = nor_get_clusters_and_centers(trans_data,formed_clusters)
  for i in range(len(Clusters)):
    for j in range(len(Clusters['Cluster' + str(i)])):
      diff = Clusters['Cluster' + str(i)][j] - Cluster_Centers['Cluster_Center' + str(i)]
      Sq_diff = (diff)**2
      sq_diff.append(Sq_diff)

  Sq_diff_sum = np.sum(np.sum(sq_diff))
  rmse = np.sqrt(Sq_diff_sum/data_nor.shape[0])
  return rmse

"""**Avg Cluster Variance**"""

def avg_var(norm_data, result):
  trans_data = pd.DataFrame(data_nor)
  trans_data['Cluster'] = result
  Clusters = {}
  Cluster_Centers = {}
  for i in set(result):
    Clusters['Cluster' + str(i)] = np.array(trans_data[trans_data.Cluster == i].drop(columns=['Cluster']))

  variances = pd.DataFrame(columns=range(len(Clusters)),index=range(2))
  for i in range(len(Clusters)):
      variances[i].iloc[0] = np.var(Clusters['Cluster' + str(i)])
      variances[i].iloc[1] = Clusters['Cluster' + str(i)].shape[0]

  var_sum = 0
  for i in range(7):
      var_sum = var_sum + (variances[i].iloc[0] * variances[i].iloc[1])

  var_avg = var_sum/data_nor.shape[0]


  return (print("The Average variance is:", var_avg))

"""**Avg Inter-cluster distance**"""

def avg_inter_dist(norm_data, clustering_results):

  from scipy.spatial.distance import cdist,pdist
  n_clusters = 7
  trans_data = pd.DataFrame(norm_data)
  trans_data['Cluster'] = clustering_results
  Clusters = {}
  Cluster_Centers = {}
  for i in set(clustering_results):
    Clusters['Cluster' + str(i)] = np.array(trans_data[trans_data.Cluster == i].drop(columns=['Cluster']))

  distance_matrix = pd.DataFrame(columns=range(len(Clusters)),index=range(len(Clusters)))
  for i in range(len(Clusters)):
    for j in range(len(Clusters)):
      if i == j:
        #distance_matrix[i].iloc[j] = 0
        distance_intra = cdist(Clusters['Cluster' + str(i)], Clusters['Cluster' + str(j)], 'euclidean')
        distance_matrix[i].iloc[j] = np.max(distance_intra)
      elif i > j:
        continue
      else:
        distance = cdist(Clusters['Cluster' + str(i)], Clusters['Cluster' + str(j)], 'euclidean')
        distance_matrix[i].iloc[j] = np.min(distance)
        distance_matrix[j].iloc[i] = np.min(distance)

  sum_min = 0
  for i in range(n_clusters):
      sum_min = sum_min + np.min(distance_matrix[i])

  avg_inter = sum_min/n_clusters

  return (print("The Average Inter-cluster dist is:", avg_inter))

"""# **Implementation**"""

data_path = '/content/multivariate-weather-data-clustering/data/ERA5_meteo_sfc_2021_daily.nc'
data = xr.open_dataset(data_path, decode_times=False)#To view the date as integers of 0, 1, 2,..
var = list(data.variables)[3:]

data_nor, data_clustering = data_preprocessing('/content/multivariate-weather-data-clustering/data/ERA5_meteo_sfc_2021_daily.nc')

data_nor.shape, data_clustering.shape

labels = parea(data_nor)

silh = silhouette_score1(data_nor,  labels)
u,indices = np.unique(labels,return_counts = True)

print("Silhouette score is ", silh)
print("Cluster index ", u, "and Cluster Sizes: ", indices)

from sklearn.metrics import davies_bouldin_score

db = davies_bouldin_score(data_nor, labels)
print("Davies-Bouldin score is ", db)

from sklearn.metrics import calinski_harabasz_score
ch = calinski_harabasz_score(data_nor, labels)
print("Davies-Bouldin score is ", ch)

print("RMSE score is ", total_rmse('/content/drive/MyDrive/Data/ERA5_meteo_sfc_2021_daily.nc', labels))

print("Variance is ", avg_var(data_nor, labels))

print("Inter-cluster distance ", avg_inter_dist(data_nor, labels))



"""#Loop multiple times"""

def best_clustering(n):

  n # Number of single models used
    #MIN_PROBABILITY = 0.6 # The minimum threshold of occurances of datapoints in a cluster

  sil = []
  sil_score = []
  # Generating a "Cluster Forest"
  #clustering_models = NUM_alg * [main()]#
  for i in range(n):
    labels_parea = parea(data_nor)
    #sil.append(results)
    #print(results)
    silhouette_avg_rdata_daily = silhouette_score1(data_nor, labels_parea)
    sil_score.append(silhouette_avg_rdata_daily)

    sil.append([silhouette_avg_rdata_daily, labels_parea])

  print("Our silhouettes range is: ", sil_score)

  max_index = 0

  for j in range(len(sil)):
    if(sil[j][0]>=sil[max_index][0]):
      max_index = j
  print(sil[max_index])

  return sil[max_index]

silh_parea, result_parea = best_clustering(20)

np.array(result_parea)

silh_parea

#Just to confirm if am reading the right array
silhouette_avg_rdata_daily = silhouette_score1(data_nor, result_parea)
print("The average silhouette_score is :", silhouette_avg_rdata_daily)

from sklearn.metrics import davies_bouldin_score

print("Davies-Bouldin score is ", davies_bouldin_score(data_nor, result_parea))

print("RMSE score is ", total_rmse('/content/drive/MyDrive/Data/ERA5_meteo_sfc_2021_daily.nc', result_parea))

print("Variance is ", avg_var(data_nor, result_parea))

print("Inter-cluster distance ", avg_inter_dist(data_nor, result_parea))

from sklearn.metrics import calinski_harabasz_score
ch = calinski_harabasz_score(data_nor, result_parea)
print("Davies-Bouldin score is ", ch)
