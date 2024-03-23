# -*- coding: utf-8 -*-
"""KMeans_ensemble.ipynb


[Source](https://www.kaggle.com/code/thedevastator/how-to-ensemble-clustering-algorithms-updated)
"""



from google.colab import drive
drive.mount('/content/drive')

"""#Install needed libraries"""

# Install dask.dataframe
!pip install "dask[dataframe]"
!pip install netCDF4

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

from collections import defaultdict
from scipy.sparse import csr_matrix
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

import seaborn as sns
import xarray as xr
#from mwdc.clustering.st_agglomerative import st_agglomerative

import warnings
warnings.filterwarnings("ignore")

# import netCDF4
# from netCDF4 import Dataset
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
# from mwdc.visualization.clusterplotting import clusterPlot2D
# from mwdc.visualization.visualization import visualization2
# from mwdc.preprocessing.preprocessing import data_preprocessing
# from mwdc.evaluation.st_evaluation import st_rmse_df, st_corr, st_rmse_np
# from mwdc.clustering.st_agglomerative import st_agglomerative

import sys
import pickle
import matplotlib as mpl
import matplotlib.colors as colors
import os
import xarray as xr
import warnings
warnings.filterwarnings("ignore")



import netCDF4 as nc
import pandas as pd
import numpy as np
import xarray as xr
import datetime
import datetime as dt
from netCDF4 import date2num,num2date
from math import sqrt
from time import time

from google.colab import drive
drive.mount('/content/drive')

from sklearn.metrics import silhouette_samples, silhouette_score
def silhouette_score1(X, labels, *, metric="cosine", sample_size=None, random_state=None, **kwds):
 return np.mean(silhouette_samples(X, labels, metric=metric, **kwds))

## This function will will pre-process our daily data for DEC model as numpy array
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
            rdata_daily_np_array_T[i, j, k, l] = overall_mean  # np.nanmean(rdata_daily_np_array_T[i, j, k, :])
  rdata_daily_np_array_T = rdata_daily_np_array_T.transpose(0,2,3,1)
  rdata_daily_np_array_T_R = rdata_daily_np_array_T.reshape((rdata_daily_np_array_T.shape[0], -1))  # transform the dailt data from (365, 7, 41, 41) to (365, 11767)
  min_max_scaler = preprocessing.MinMaxScaler() # calling the function
  rdata_daily_np_array_T_R_nor = min_max_scaler.fit_transform(rdata_daily_np_array_T_R)   # now normalize the data, otherwise the loss will be very big
  #rdata_daily_np_array_T_R_nor = np.float32(rdata_daily_np_array_T_R_nor)    # convert the data type to float32, otherwise the loass will be out-of-limit
  rdata_daily_np_array_T_R_nor_R = rdata_daily_np_array_T_R_nor.reshape((rdata_daily_np_array_T_R_nor.shape[0], rdata_daily_np_array.shape[2], rdata_daily_np_array.shape[3], rdata_daily_np_array.shape[0]))
  return rdata_daily_np_array_T_R_nor, rdata_daily_np_array_T_R_nor_R

path = '/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/'

data_nor_eval, data_clustering = data_preprocessing('/content/drive/MyDrive/Data/ERA5_meteo_sfc_2021_daily.nc')

data_nor_eval.shape, data_clustering.shape

df = data_nor_eval

def round_2_dec(X):
  for i in X:
    data2 = X.round(2)
    #print(data2)
  return data2

from sklearn.metrics import silhouette_samples, silhouette_score
def silhouette_score1(X, labels, *, metric="cosine", sample_size=None, random_state=None, **kwds):
 return np.mean(silhouette_samples(X, labels, metric=metric, **kwds))

def pca1(data,n): # data is data to be input , n is the number of components
  pca = PCA(n_components=n)
  pca.fit(data)

  # Get pca scores
  pca_scores = pca.transform(data)

  # Convert pca_scores to a dataframe
  scores_df = pd.DataFrame(pca_scores)

  # Round to two decimals
  scores_df = scores_df.round(2)

  # Return scores
  return scores_df

"""After fitting and transforming data, we will display Kullback-Leibler (KL) divergence between the high-dimensional probability distribution and the low-dimensional probability distribution.

**Low KL divergence is a sign of better results.**
"""

def best_clustering(n):

  n # Number of single models used
    #MIN_PROBABILITY = 0.6 # The minimum threshold of occurances of datapoints in a cluster

  sil = []
  sil_score = []
  # Generating a "Cluster Forest"
  #clustering_models = NUM_alg * [main()]#
  for i in range(n):
    results = KMeans(n_clusters = 7).fit_predict(df)
    #results = main()
    #sil.append(results)
    #print(results)
    silhouette_avg_rdata_daily = silhouette_score1(df, results)
    sil_score.append(silhouette_avg_rdata_daily)

    sil.append([silhouette_avg_rdata_daily, results])

  print("Our silhouettes range is: ", sil_score)

  max_index = 0

  for j in range(len(sil)):
    if(sil[j][0]>=sil[max_index][0]):
      max_index = j
  print(sil[max_index])

  return sil[max_index]

silhouette, result = best_clustering(20)

nor_data = df

silhouette_avg_rdata_daily = silhouette_score1(nor_data, result)
print("The average silhouette_score is :", silhouette_avg_rdata_daily)

from sklearn.metrics import davies_bouldin_score

print("Davies-Bouldin score is ", davies_bouldin_score(nor_data, result))

"""# **4. Evaluation:**
To compute the RMSE, variance, and average inter cluster distance we have to use the array format of our dataset and the clustering result.
"""

def total_rmse(data_path,formed_clusters):
  # processed_data = data_preprocessing(data_path)
  processed_data = nor_data
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
  rmse = np.sqrt(Sq_diff_sum/nor_data.shape[0])
  return rmse

total_rmse('/content/drive/MyDrive/Data/ERA5_meteo_sfc_2021_daily.nc', result)

"""### This cell measure the variances of the generated clusters."""

trans_data = pd.DataFrame(nor_data)
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

var_avg = var_sum/nor_data.shape[0]
var_avg

"""### The following cell measure the average inter cluster distance."""

from scipy.spatial.distance import cdist,pdist

trans_data = pd.DataFrame(nor_data)
trans_data['Cluster'] = result
Clusters = {}
Cluster_Centers = {}
for i in set(result):
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
for i in range(len(Clusters)):
    sum_min = sum_min + np.min(distance_matrix[i])

avg_inter = sum_min/len(Clusters)
avg_inter

silh = silhouette_score1(data_nor_eval,  result)
u,indices = np.unique(result,return_counts = True) # sc=0.3412 st 64
# u,indices
print(silh)
print(u,indices)

silhouette, result_1 = best_clustering(20)

silh = silhouette_score1(data_nor_eval,  result_1)
u,indices = np.unique(result_1,return_counts = True) # sc=0.3412 st 64
# u,indices
print(silh)
print(u,indices)



silhouette, result_2 = best_clustering(20)

silh = silhouette_score1(data_nor_eval,  result_2)
u,indices = np.unique(result_2,return_counts = True) # sc=0.3412 st 64
# u,indices
print(silh)
print(u,indices)

silhouette, result_3 = best_clustering(20)

silh = silhouette_score1(data_nor_eval,  result_3)
u,indices = np.unique(result_3,return_counts = True) # sc=0.3412 st 64
# u,indices
print(silh)
print(u,indices)

#pickle.dump(result_3, open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/KMeans_0.3125.pkl", "wb"))

silhouette, result_4 = best_clustering(20)

silh = silhouette_score1(data_nor_eval,  result_4)
u,indices = np.unique(result_4,return_counts = True) # sc=0.3412 st 64
# u,indices
print(silh)
print(u,indices)

#pickle.dump(result_4, open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/KMeans_0.3389.pkl", "wb"))

silhouette, result_5 = best_clustering(20)

silh = silhouette_score1(data_nor_eval,  result_5)
u,indices = np.unique(result_5,return_counts = True) # sc=0.3412 st 64
# u,indices
print(silh)
print(u,indices)

clusters_list = [result, result_1,result_2, result_3, result_4, result_5]
kmeans_ens = [result_3, result_4, result_5]#result, result_1,result_2,





class ClusterSimilarityMatrix():

    def __init__(self) -> None:
        self._is_fitted = False

    def fit(self, y_clusters):
        if not self._is_fitted:
            self._is_fitted = True
            self.similarity = self.to_binary_matrix(y_clusters)
            return self

        self.similarity += self.to_binary_matrix(y_clusters)

    def to_binary_matrix(self, y_clusters):
        y_reshaped = np.expand_dims(y_clusters, axis=-1)
        return (cdist(y_reshaped, y_reshaped, 'cityblock')==0).astype(int)

NUM_alg = 100
occurrence_threshold = 0.45

clustering_models = NUM_alg * [result, result_1, result_2, result_3, result_4, result_5]


clt_sim_matrix = ClusterSimilarityMatrix() #Initializing the similarity matrix
for model in clustering_models:
  clt_sim_matrix.fit(model)

sim_matrixx = clt_sim_matrix.similarity
sim_matrixx = sim_matrixx/sim_matrixx.diagonal()
sim_matrixx[sim_matrixx < occurrence_threshold] = 0

unique_labels = np.unique(np.concatenate(clustering_models))

print(sim_matrixx)
np.save('/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/KMeans_co_occurrence_matrix.npy', sim_matrixx)
#print(norm_sim_matrix)

import numpy as geek

data_nor = data_nor_eval

#sim_matrixx = geek.load('/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Non-negative Matrix Factorization/DEC_co_occurrence_matrix.npy')

from sklearn.cluster import SpectralClustering
spec_clt = SpectralClustering(n_clusters=7, affinity='precomputed',
                              n_init=5, random_state=214)
final_labels = spec_clt.fit_predict(sim_matrixx)

final_labels





#pickle.dump(final_labels, open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/DSC_Ensemble1_032.pkl", "wb"))

result = pickle.load(open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/DSC_Ensemble1_032.pkl", "rb"))

data_nor_eval = data_nor

"""# **Evaluation Metrics**

**Silhouette Score**
"""

def silhouette_score1(X, labels, *, metric="cosine", sample_size=None, random_state=None, **kwds):
 return np.mean(silhouette_samples(X, labels, metric=metric, **kwds))

"""**Davies bouldin**"""

def davies_bouldin_score(X, labels):
 return print("Davies-Bouldin score is ", davies_bouldin_score(X, labels))

"""**Calinski Harabasz**"""

def calinski_harabasz_score(X, labels):
 return print("Calinski Harabasz score is ", calinski_harabasz_score(X, labels))

"""**RMSE**"""

data_nor = df

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

silh = silhouette_score1(data_nor,  final_labels)
u,indices = np.unique(final_labels,return_counts = True) # sc=0.3412 st 64
# u,indices
print(silh)
print(u,indices)

pickle.dump(final_labels, open(path + 'kmeans_fin_ens_' + str(silh) + '.pkl', "wb"))

silh = silhouette_score1(data_nor,  final_labels)
u,indices = np.unique(final_labels,return_counts = True)

print("Silhouette score is ", silh)
print("Cluster index ", u, "and Cluster Sizes: ", indices)

from sklearn.metrics import davies_bouldin_score

db = davies_bouldin_score(data_nor, final_labels)
print("Davies-Bouldin score is ", db)

from sklearn.metrics import calinski_harabasz_score
ch = calinski_harabasz_score(data_nor, final_labels)
print("Davies-Bouldin score is ", ch)

print("RMSE score is ", total_rmse('/content/drive/MyDrive/Data/ERA5_meteo_sfc_2021_daily.nc', final_labels))

print("Variance is ", avg_var(data_nor, final_labels))

print("Inter-cluster distance ", avg_inter_dist(data_nor, final_labels))



"""#Loop multiple times"""

def best_clustering(n):

  n # Number of single models used
    #MIN_PROBABILITY = 0.6 # The minimum threshold of occurances of datapoints in a cluster

  sil = []
  sil_score = []
  # Generating a "Cluster Forest"
  #clustering_models = NUM_alg * [main()]#
  for i in range(n):
    spec_clt = SpectralClustering(n_clusters=7, affinity='precomputed',
                              n_init=5, random_state=214)
    final_labels = spec_clt.fit_predict(sim_matrixx)
    silhouette_avg_rdata_daily = silhouette_score1(data_nor, final_labels)
    sil_score.append(silhouette_avg_rdata_daily)

    sil.append([silhouette_avg_rdata_daily, final_labels])

  print("Our silhouettes range is: ", sil_score)

  max_index = 0

  for j in range(len(sil)):
    if(sil[j][0]>=sil[max_index][0]):
      max_index = j
  print(sil[max_index])

  return sil[max_index]

silh_ce, result_ce = best_clustering(20)

#Just to confirm if am reading the right array
silhouette_avg_rdata_daily = silhouette_score1(data_nor, result_ce)
print("The average silhouette_score is :", silhouette_avg_rdata_daily)

from sklearn.metrics import davies_bouldin_score

print("Davies-Bouldin score is ", davies_bouldin_score(data_nor, result_ce))

from sklearn.metrics import calinski_harabasz_score
ch = calinski_harabasz_score(data_nor, result_ce)
print("Davies-Bouldin score is ", ch)

print("RMSE score is ", total_rmse('/content/drive/MyDrive/Data/ERA5_meteo_sfc_2021_daily.nc', result_ce))

print("Variance is ", avg_var(data_nor, result_ce))

print("Inter-cluster distance ", avg_inter_dist(data_nor, result_ce))















class ClusterSimilarityMatrix():

    def __init__(self) -> None:
        self._is_fitted = False

    def fit(self, y_clusters):
        if not self._is_fitted:
            self._is_fitted = True
            self.similarity = self.to_binary_matrix(y_clusters)
            return self

        self.similarity += self.to_binary_matrix(y_clusters)

    def to_binary_matrix(self, y_clusters):
        y_reshaped = np.expand_dims(y_clusters, axis=-1)
        return (cdist(y_reshaped, y_reshaped, 'cityblock')==0).astype(int)

NUM_alg = 50
occurrence_threshold = 0.4

clusters_list = NUM_alg * [result, result_1, result_2, result_3, result_4, result_5]

kmeans_ens = NUM_alg * [result_3, result_4, result_5]

clt_sim_matrix = ClusterSimilarityMatrix() #Initializing the similarity matrix
for model in clusters_list:
  clt_sim_matrix.fit(model)

sim_matrixx = clt_sim_matrix.similarity
sim_matrixx = sim_matrixx/sim_matrixx.diagonal()
sim_matrixx[sim_matrixx < occurrence_threshold] = 0

unique_labels = np.unique(np.concatenate(clusters_list))

print(sim_matrixx)

#below is for the final ensemble clustering
km_clt_sim_matrix = ClusterSimilarityMatrix() #Initializing the similarity matrix
for model in kmeans_ens:
  km_clt_sim_matrix.fit(model)

km_sim_matrixx = km_clt_sim_matrix.similarity
km_sim_matrixx = km_sim_matrixx/km_sim_matrixx.diagonal()
km_sim_matrixx[km_sim_matrixx < occurrence_threshold] = 0

km_unique_labels = np.unique(np.concatenate(kmeans_ens))

# print(km_sim_matrixx)

#np.save('/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Non-negative Matrix Factorization/KMeans_ens_co_occurrence_matrix.npy', sim_matrixx)
#print(norm_sim_matrix)

def best_clustering5(n):

  n # Number of single models used
    #MIN_PROBABILITY = 0.6 # The minimum threshold of occurances of datapoints in a cluster

  sil = []
  sil_score = []
  # Generating a "Cluster Forest"
  #clustering_models = NUM_alg * [main()]#
  for i in range(n):
    # model = SpectralClustering(n_clusters=7, affinity='nearest_neighbors', random_state=0)
    # # fit model and predict clusters
    # yhat_sc = model.fit_predict(sim_matrixx)

    yhat_sc = KMeans(n_clusters = 7).fit_predict(df)

    silhouette_avg_rdata_daily = silhouette_score1(df, yhat_sc)
    sil_score.append(silhouette_avg_rdata_daily)

    sil.append([silhouette_avg_rdata_daily, yhat_sc])

  print("Our silhouettes range is: ", sil_score)

  max_index = 0

  for j in range(len(sil)):
    if(sil[j][0]>=sil[max_index][0]):
      max_index = j
  print(sil[max_index])

  #return sil[max_index]
  return sil[max_index]

silhouette, result_sc = best_clustering5(20)

silh = silhouette_score1(data_nor_eval,  result_sc)
u,indices = np.unique(result_sc,return_counts = True) # sc=0.3412 st 64
# u,indices
print(silh)
print(u,indices)

def best_clustering(n):

  n # Number of single models used
    #MIN_PROBABILITY = 0.6 # The minimum threshold of occurances of datapoints in a cluster

  sil = []
  sil_score = []
  # Generating a "Cluster Forest"
  #clustering_models = NUM_alg * [main()]#
  for i in range(n):
    results_en = KMeans(n_clusters = 7).fit_predict(sim_matrixx)
    #results = main()
    #sil.append(results)
    #print(results)
    silhouette_avg_rdata_daily = silhouette_score1(df, results_en)
    sil_score.append(silhouette_avg_rdata_daily)

    sil.append([silhouette_avg_rdata_daily, results_en])

  print("Our silhouettes range is: ", sil_score)

  max_index = 0

  for j in range(len(sil)):
    if(sil[j][0]>=sil[max_index][0]):
      max_index = j
  print(sil[max_index])

  return sil[max_index]



silhouette, result_sc = best_clustering5(20)

silh = silhouette_score1(data_nor_eval,  result_sc)
u,indices = np.unique(result_sc,return_counts = True) # sc=0.3412 st 64
# u,indices
print(silh)
print(u,indices)

nor_data = data_nor_eval

from sklearn.metrics import davies_bouldin_score

print("Davies-Bouldin score is ", davies_bouldin_score(nor_data, result_sc))

"""# **4. Evaluation:**
To compute the RMSE, variance, and average inter cluster distance we have to use the array format of our dataset and the clustering result.
"""

def total_rmse(data_path,formed_clusters):
  # processed_data = data_preprocessing(data_path)
  processed_data = nor_data
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
  rmse = np.sqrt(Sq_diff_sum/nor_data.shape[0])
  return rmse

total_rmse('/content/drive/MyDrive/Data/ERA5_meteo_sfc_2021_daily.nc', result_sc)

"""### This cell measure the variances of the generated clusters."""

trans_data = pd.DataFrame(nor_data)
trans_data['Cluster'] = result_sc
Clusters = {}
Cluster_Centers = {}
for i in set(result_sc):
  Clusters['Cluster' + str(i)] = np.array(trans_data[trans_data.Cluster == i].drop(columns=['Cluster']))

variances = pd.DataFrame(columns=range(len(Clusters)),index=range(2))
for i in range(len(Clusters)):
    variances[i].iloc[0] = np.var(Clusters['Cluster' + str(i)])
    variances[i].iloc[1] = Clusters['Cluster' + str(i)].shape[0]

var_sum = 0
for i in range(7):
    var_sum = var_sum + (variances[i].iloc[0] * variances[i].iloc[1])

var_avg = var_sum/nor_data.shape[0]
var_avg

"""### The following cell measure the average inter cluster distance."""

from scipy.spatial.distance import cdist,pdist

trans_data = pd.DataFrame(nor_data)
trans_data['Cluster'] = result_sc
Clusters = {}
Cluster_Centers = {}
for i in set(result_sc):
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
for i in range(len(Clusters)):
    sum_min = sum_min + np.min(distance_matrix[i])

avg_inter = sum_min/len(Clusters)
avg_inter

"""# **Second Approach**"""

# print("The average silhouette_score is :", silhouette)
# print("The result :", result)

# silhouette_avg_rdata_daily_last = 0
# result_last = 0
# for ite in range(int(20)):
#   results = KMeans(n_clusters = 7).fit_predict(df)
#   #result = train()
#   silhouette_avg_rdata_daily = silhouette_score1(data_nor_eval, result)
#   print("The average silhouette_score is :", silhouette_avg_rdata_daily)
#   if silhouette_avg_rdata_daily_last < silhouette_avg_rdata_daily:
#     silhouette_avg_rdata_daily_last = silhouette_avg_rdata_daily
#     #print("The average silhouette_score is :", silhouette_avg_rdata_daily_last)
#     result_last = result
#     print("The result :", result_last)

# print("The average silhouette_score is :", silhouette_avg_rdata_daily_last)
# print("The result :", result_last)

cls_tup_list = []
for cls_tup in zip(*clusters_list):
    cls_tup_list.append(cls_tup)
zipper = {x: i for i, x in enumerate(sorted(set(cls_tup_list)))}
zipped_list = [zipper[x] for x in cls_tup_list]
unzipper = defaultdict(set)
for idx, cls_tup in enumerate(cls_tup_list):
    zipped = zipper[cls_tup]
    unzipper[zipped].add(idx)
comp_clusters_list = [[-1]*len(zipper) for _ in range(len(clusters_list))]
for clusters, comp_clusters in zip(clusters_list, comp_clusters_list):
    for i, cluster_i in enumerate(clusters):
            value = zipped_list[i]
            comp_clusters[value] = cluster_i

from tqdm import trange

def create_sparse_matrix(clusters):
    n = len(clusters)
    data = []
    row = []
    col = []
    # O(n**2)
    for i in trange(n):
        for j in range(i+1, n):
            if clusters[i] == clusters[j]:
                data.append(1)
                row.append(i)
                col.append(j)
    return csr_matrix((data, (row, col)), shape=(n, n))

sparse_matrix_list = [create_sparse_matrix(comp_clusters) for comp_clusters in comp_clusters_list]

sparse_matrix_mean = (sparse_matrix_list[0] * 0.5 + sparse_matrix_list[1] * 0.5)

# The lower the threshold the lower number of clusters: Since we are counting by the connected component.
# It seems desirable to set it a little higher.
threshold = 0.2
sparse_matrix_mean[sparse_matrix_mean < threshold] = 0

"""#Now we reconstruct our clusters and return an integer column#"""

sparse_matrix_mean = sparse_matrix_mean.toarray()
sparse_matrix_mean

"""## **Intuition:** The clusters are combined in order of increasing sparse_matrix_mean[node1][node2].

Disjoint Set Union(DSU) is useful, with complexity of O(α(N)).

Algorithm: Repeat until the size of each cluster reaches SIZ_MAX or the number of clusters reaches CLS_NUM_MIN. This is fast enough: O(log(N)*N).##
"""

# Sort sparse_matrix_mean[node1][node2] that are non-zero.
# It is fast enough because the number of edges is small due to preprocessing (zipper).
clusters_final = np.zeros(len(df))
clusters_final_next_id = 0

node_end = len(comp_clusters_list[0])
edge_list = []  # [(w, fr, to), ...]
for fr in range(node_end):
    for to in range(fr, node_end):
        w = sparse_matrix_mean[fr][to]
        if w == 0:
            continue
        edge_list.append((w, fr, to))
edge_list.sort(reverse=True)

SIZ_MAX = 18000
CLS_NUM_MIN = 7

"""**Disjoint Set Union(DSU)**"""

class DSU:
    def __init__(self, node_end, unzipper):
        self.par = [i for i in range(node_end)]
        self.siz = [len(unzipper[i]) for i in range(node_end)]
        self.cls_num = node_end  # To use the cls_num variable, we declared this class

    def find(self, x):
        if self.par[x] == x: return x
        self.par[x] = self.find(self.par[x])
        return self.par[x]

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)
        if x == y:
            return
        if self.siz[x] > self.siz[y]: x, y = y, x
        self.par[x] = y
        self.siz[y] += self.siz[x]
        self.cls_num -= 1

    def get_siz(self, x):
        x = self.find(x)
        return self.siz[x]

dsu = DSU(node_end, unzipper)
for w, fr, to in edge_list:
    if (dsu.get_siz(fr)+dsu.get_siz(to)) > SIZ_MAX: continue
    dsu.union(fr, to)
    if dsu.cls_num <= CLS_NUM_MIN:
        print("number of clusters reaches CLS_NUM_MIN: {}".format(CLS_NUM_MIN), " break.")
        break

clusters_final = [0]*len(clusters_list[0])
for node in range(node_end):
    cluster_id = dsu.find(node)
    idx_list = unzipper[node]
    for idx in idx_list:
        clusters_final[idx] = cluster_id

zipper = {x: i for i, x in enumerate(sorted(set(clusters_final)))}
clusters_final = [zipper[x] for x in clusters_final]

silhouette_avg_rdata_daily = silhouette_score1(data_nor_eval, clusters)
print("The average silhouette_score is :", silhouette_avg_rdata_daily)

# # https://www.kaggle.com/code/ambrosm/tpsjul22-gaussian-mixture-cluster-analysis
# def compare_clusterings(y1, y2, title=''):
#     """Show the adjusted rand score and plot the two clusterings in color"""
#     ars = adjusted_rand_score(y1, y2)
#     n1 = y1.max() + 1
#     n2 = y2.max() + 1
#     argsort = np.argsort(y1*100 + y2) if n1 >= n2 else np.argsort(y2*100 + y1)
#     plt.figure(figsize=(16, 0.5))
#     for i in range(6, 11):
#         plt.scatter(np.arange(len(y1)), np.full_like(y1, i), c=y1[argsort], s=1, cmap='tab10')
#     for i in range(5):
#         plt.scatter(np.arange(len(y2)), np.full_like(y2, i), c=y2[argsort], s=1, cmap='tab10')
#     plt.gca().axis('off')
#     plt.title(f'{title}\nAdjusted Rand score: {ars:.5f}')
#     plt.savefig(title + '.png', bbox_inches='tight')
#     plt.show()

# for clusters in clusters_list: compare_clusterings(np.array(clusters), np.array(clusters_final))

df = pd.read_csv("/content/drive/MyDrive/ECRP_Data_Science/Implementation and Results/Jianwu_test/PCA_Result_Combination/PCA_Combined_var1.csv")

df2 = df.time_step
df2

df1 = data_nor_eval
# creating pandas dataframe on transformed data
df1 = pd.DataFrame(df2, columns=['time_step'])
df1['clusterid'] =  clusters
#df1["cluster"] = cluster.labels_
df1['clusterid'].value_counts()
df1.to_csv("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/K-Means.csv")
df1

df1.groupby('clusterid').count()

ensemble = silhouette_score1(data_nor_eval,  clusters)
u,indices = np.unique(clusters,return_counts = True) # sc=0.3412 st 64
# u,indices
print(ensemble)
print(u,indices)



import pickle

#pickle.dump(clusters, open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/KMeans_Ensemble1_029.pkl", "wb"))

#cluster_result = pickle.load(open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/KMeans_Ensemble1_029.pkl", "rb"))

base_clustering = silhouette_score1(data_nor_eval,  clusters)

base_clustering

base_clustering = silhouette_score1(data_nor_eval,  clusters)
base_clustering

u,indices = np.unique(clusters,return_counts = True) # sc=0.3412 st 64
u,indices

np.array(clusters)

# top_10_sc = []
# top_10_results = []
# lowest_ss = 0
# result_last = 0
# count = 0
# def add_result(silhouette_s, result):
#   num_res = len(top_10_sc)
#   print("The average silhouette_score is to add :", silhouette_s)
#   if num_res == 10:
#     minpos = top_10_sc.index(min(top_10_sc))
#     print("The value removed from the list :", min(top_10_sc))
#     top_10_sc.pop(minpos)
#     top_10_results.pop(minpos)
#   top_10_sc.append(silhouette_s)
#   top_10_results.append(result)
#   return min(top_10_sc)


# for ite in range(int(20)):
#   result = KMeans(n_clusters = 7).fit_predict(data_nor_eval)
#   #result = train()
#   silhouette_s = silhouette_score1(data_nor_eval, result)
#   print("The average silhouette_score is :", silhouette_s)
#   if count < 11:
#     lowest_ss = add_result(silhouette_s, result)
#     count = count + 1
#   else:
#     if silhouette_s > lowest_ss:
#       lowest_ss = add_result(silhouette_s, result)
# top_10_sc

# data_nor_eval

# add_result(silhouette_s, result)





# /content/DAC_models/DAC_model_final_1700598391.ckpt.data-00000-of-00001

# /content/DAC_models/DAC_model_final_1700598391.ckpt.index
