# -*- coding: utf-8 -*-
"""Ensemble_SC.ipynb


Paper: Ensemble Learning for Spectral Clustering Implementation of the paper “Li, H., Ye, X., Imakura, A. and Sakurai, T., 2020, November. Ensemble learning for spectral clustering. In 2020 IEEE International Conference on Data Mining (ICDM) (pp. 1094-1099). IEEE” In Python.



Code Source: https://github.com/MasoudJTehrani/Ensemble_SC/tree/main
"""

from google.colab import drive
drive.mount('/content/drive')

"""#Install needed libraries"""

# Install dask.dataframe
!pip install "dask[dataframe]"
!pip install netCDF4

!git clone https://

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

# import netCDF4
# from netCDF4 import Dataset
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
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

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plotter
from scipy import stats
from numpy import linalg
from sklearn import datasets
from scipy.io import loadmat
from scipy.sparse import csgraph
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.neighbors import kneighbors_graph
from scipy.optimize import linear_sum_assignment

"""# Importing Datasets"""

# def mat2pd(file_mat, revert = False):
#     mat = loadmat(file_mat)

#     cols = len(mat['fea'])
#     rows = len(mat['fea'][0])
#     data = []
#     if (revert):
#       cols,rows = rows,cols
#       for row in range(rows):
#         a_row = []
#         for col in range(cols):
#           a_row.append(mat['fea'][row][col])
#         a_row.append(mat['gnd'][row][0])
#         data.append(a_row)
#       df = pd.DataFrame(data)
#       return(df)
#     else:
#       for row in range(rows):
#         a_row = []
#         for col in range(cols):
#           a_row.append(mat['fea'][col][row])
#         a_row.append(mat['gnd'][row][0])
#         data.append(a_row)
#       df = pd.DataFrame(data)
#       return(df)

# iris = mat2pd('iris_uni.mat')

# jaffe = mat2pd('jaffe.mat',True)

# pixraw = mat2pd('pixraw10P.mat')

# wine = mat2pd('wine_uni.mat')

# warp = mat2pd('warpPIE10P.mat')

# halfcircles = mat2pd('halfcircles.mat')

# threecircles = mat2pd('threecircles.mat')

"""# Functions"""

def visualize_dataset(title, data, labels):
    plotter.scatter(data[:, 0], data[:, 1], c=labels)
    plotter.title(title)
    plotter.show()

def initialize_L(laplacians):
    L = np.zeros(laplacians[0].shape)
    for i in range(len(laplacians)):
        L += laplacians[i]

    return L


def calculate_U(L, clusters):
    U, sigma, _ = linalg.svd(L)
    return U[:, 0: clusters]


def update_L(U_list):
    L = np.zeros((U_list[0].shape[0], U_list[0].shape[0]))
    for i in range(len(U_list)):
        L += np.matmul(U_list[i], np.transpose(U_list[i]))

    return L


def calculate_obj(U, L):
    return np.trace(np.matmul(np.transpose(U), np.matmul(L, U)))


def update_alpha(alpha_0, objs):
    return alpha_0 * np.power(np.std(objs), 2)

def gaussian(dataset):
    gamma = 0.1
    landa = 1/(2*(gamma**2))
    result = []
    for row1 in dataset:
      row = []
      for row2 in dataset:
        row.append(np.exp(landa * squared_euclidean_distance(row1, row2)))
      result.append(row)
    return result


def squared_euclidean_distance(data1, data2):
    subtract = data1 - data2;
    return np.matmul(subtract, subtract)


def euclidean_distance(data1, data2):
    return np.sqrt(squared_euclidean_distance(data1, data2))

def k_means_clustering(dataset, clusters, rng):
    k_means = KMeans(n_clusters=clusters, random_state=rng)
    return k_means.fit_predict(dataset)


def spectral_clustering(dataset, rng, no_clusters, k = 5):
    #clusters = len(np.unique(dataset.iloc[:, -1].values))
    clusters = no_clusters
    dataset = dataset.iloc[:, :-1]
    L = compute_laplacian(dataset, k, 0.5, 'e')
    U = calculate_U(L, clusters)
    return k_means_clustering(U, clusters, rng)


def elsc_clustering(dataset, rng , no_clusters, methods = ['e', 's', 'p'], a = 2, b = 9):
    #clusters = len(np.unique(dataset.iloc[:, -1].values))
    clusters = no_clusters
    #dataset = dataset.iloc[:, :-1]
    alpha = 0.001
    input_laplacians = generate_input_laplacians(dataset, methods, a, b)

    L = initialize_L(input_laplacians)
    U = calculate_U(L, clusters)

    U_list = []
    objs = []
    alpha_0 = alpha

    for i in range(len(input_laplacians)):
        U_list.append(calculate_U(input_laplacians[i], clusters))
        objs.append(calculate_obj(U_list[i], input_laplacians[i]))
        alpha = update_alpha(alpha_0, objs)

    counter = 0
    while counter < 10:
        temp = alpha * np.matmul(U, np.transpose(U))
        for i in range(len(input_laplacians)):
            modified_L = input_laplacians[i] + temp
            U_list[i] = calculate_U(modified_L, clusters)
            objs[i] = calculate_obj(U_list[i], modified_L)
            alpha = update_alpha(alpha_0, objs)

        L = update_L(U_list)
        U = calculate_U(L, clusters)

        counter += 1

    for row in range(U.shape[0]):
      temp = 0
      for col in range(U.shape[1]):
        temp += U[row, col] ** 2
      U[row, :] = U[row, :] / np.sqrt(temp)

    return k_means_clustering(U, clusters, rng)

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

"""## Laplacian Functions"""

def generate_input_laplacians(dataset, methods = ['e', 's', 'p'], a = 2, b = 9):
    laplacians = [ compute_laplacian(dataset, k, 0.5, m) for k in range(a,b) for m in methods ]
    return laplacians

def compute_kernel(D, k, coef):
  data_count, feature_count = D.shape

  dump = np.zeros((data_count, k))

  for i in range(data_count):
    indices = np.argpartition(D[i, :], k)
    dump[i, :] = D[i, indices[: k]]
    D[i, indices[: k]] = float('inf')

  sigma = np.mean(dump) * coef

  dump = np.exp(-dump / (2 * sigma ** 2))

  sumD = np.sum(dump, axis=1) + 1e-10;

  for i in range(data_count):
    dump[i, :] /= sumD[i]

  kernel = np.zeros(D.shape)

  for i in range(data_count):
    index = 0
    for j in range(feature_count):
      if D[i, j] == float('inf'):
        kernel[i, j] = dump[i, index]
        index += 1

  kernel = (kernel + np.transpose(kernel)) / 2

  return kernel


def compute_laplacian(dataset, k, coef, method):
  if (method == 'e'):
    dis = squareform(pdist(dataset))
  elif (method == 's'):
    rho , pval = stats.spearmanr(dataset, axis=1)
    dis = rho
  else:
    r = np.corrcoef(dataset)
    dis = 1 - r

  kernel = compute_kernel(dis, k, coef)
  sums = kernel.sum(axis=1)
  sums = np.sqrt(1 / sums)
  M = np.multiply(sums[np.newaxis, :], np.multiply(kernel, sums[:, np.newaxis]))
  return M

"""# Testing

## synthetic datasets
"""

# rng = 5

# labels1 = k_means_clustering(dataset=halfcircles.iloc[:, :-1].values , clusters = len(np.unique(halfcircles.iloc[:, -1])),  rng=rng)
# labels2 = spectral_clustering(dataset=halfcircles, rng=rng, k = 15)

# visualize_dataset("Main Data", halfcircles.iloc[:, :-1].values, halfcircles.iloc[:, -1].values)
# visualize_dataset("Kmeans_Clustering", halfcircles.iloc[:, :-1].values, labels1)
# visualize_dataset("Spectral_Clustering", halfcircles.iloc[:, :-1].values, labels2)

# labels1 = k_means_clustering(dataset=threecircles.iloc[:, :-1].values , clusters = len(np.unique(threecircles.iloc[:, -1])),  rng=rng)
# labels2 = spectral_clustering(dataset=threecircles, rng=rng, k = 14)

# visualize_dataset("Main Data", threecircles.iloc[:, :-1].values, threecircles.iloc[:, -1].values)
# visualize_dataset("Kmeans_Clustering", threecircles.iloc[:, :-1].values, labels1)
# visualize_dataset("Spectral_Clustering", threecircles.iloc[:, :-1].values, labels2)

"""## real datasets"""

# ds = iris
# labels = elsc_clustering(dataset=ds , rng=rng)
# iris_ELSC = round(cluster_acc(labels, ds.iloc[:, -1].values) * 100, 2)
# labels = spectral_clustering(dataset=ds, rng=rng)
# iris_sc = round(cluster_acc(labels, ds.iloc[:, -1].values) * 100, 2)

# ds = jaffe
# labels = elsc_clustering(dataset=ds , rng=rng)
# jaffe_ELSC = round(cluster_acc(labels, ds.iloc[:, -1].values) * 100, 2)
# labels = spectral_clustering(dataset=ds, rng=rng)
# jaffe_sc = round(cluster_acc(labels, ds.iloc[:, -1].values) * 100, 2)

# ds = pixraw
# labels = elsc_clustering(dataset=ds , rng=rng)
# pixraw_ELSC = round(cluster_acc(labels, ds.iloc[:, -1].values) * 100, 2)
# labels = spectral_clustering(dataset=ds, rng=rng)
# pixraw_sc = round(cluster_acc(labels, ds.iloc[:, -1].values) * 100, 2)

# ds = wine
# labels = elsc_clustering(dataset=ds , rng=rng)
# wine_ELSC = round(cluster_acc(labels, ds.iloc[:, -1].values) * 100, 2)
# labels = spectral_clustering(dataset=ds, rng=rng , k = 7)
# wine_sc = round(cluster_acc(labels, ds.iloc[:, -1].values) * 100, 2)

# ds = warp
# labels = elsc_clustering(dataset=ds , rng=rng, a = 2, b = 10)
# warp_ELSC = round(cluster_acc(labels, ds.iloc[:, -1].values) * 100, 2)
# labels = spectral_clustering(dataset=ds, rng=rng)
# warp_sc = round(cluster_acc(labels, ds.iloc[:, -1].values) * 100, 2)

# print("\t\t\t  SC\t\t\t  ELSC")
# print("-"*85)
# print("iris:\t\t\t", iris_sc," %\t\t", iris_ELSC," %")
# print("jaffe:\t\t\t", jaffe_sc,"%\t\t", jaffe_ELSC,"%")
# print("pixraw:\t\t\t", pixraw_sc," %\t\t", pixraw_ELSC," %")
# print("wine:\t\t\t", wine_sc,"%\t\t", wine_ELSC," %")
# print("warp:\t\t\t", warp_sc,"%\t\t", warp_ELSC,"%")



! pip install netCDF4

import netCDF4 as nc
import pandas as pd
import numpy as np
import xarray as xr
import datetime
import datetime as dt
from netCDF4 import date2num,num2date
from math import sqrt

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

rng = 30
ds = pd.DataFrame(data_nor)
labels = elsc_clustering(dataset=ds , rng=rng, no_clusters=7)
#iris_ELSC = round(cluster_acc(labels, ds.iloc[:, -1].values) * 100, 2)
# labels_1 = spectral_clustering(dataset=ds, rng=rng, no_clusters=7)
#iris_sc = round(cluster_acc(labels, ds.iloc[:, -1].values) * 100, 2)

labels

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



RMSE = st_rmse_np(data_path,labels)
RMSE

diagonal = np.diagonal(RMSE)
mean_diagonal = np.mean(diagonal)

print(mean_diagonal)

spatial_correl = st_corr(data_path, var, labels, transformation=True)
spatial_correl

diagonal = np.diagonal(spatial_correl)
mean_diagonal = np.mean(diagonal)

print(mean_diagonal)

"""#Loop multiple times"""

def best_clustering(n):

  n # Number of single models used
    #MIN_PROBABILITY = 0.6 # The minimum threshold of occurances of datapoints in a cluster

  sil = []
  sil_score = []
  # Generating a "Cluster Forest"
  #clustering_models = NUM_alg * [main()]#
  for i in range(n):
    labels_sc = elsc_clustering(dataset=ds , rng=rng, no_clusters=7)
    #sil.append(results)
    #print(results)
    silhouette_avg_rdata_daily = silhouette_score1(data_nor, labels_sc)
    sil_score.append(silhouette_avg_rdata_daily)

    sil.append([silhouette_avg_rdata_daily, labels_sc])

  print("Our silhouettes range is: ", sil_score)

  max_index = 0

  for j in range(len(sil)):
    if(sil[j][0]>=sil[max_index][0]):
      max_index = j
  print(sil[max_index])

  return sil[max_index]

silh_sc, result_sc = best_clustering(20)

result_sc

silh_sc

from sklearn.metrics import davies_bouldin_score

print("Davies-Bouldin score is ", davies_bouldin_score(data_nor, result_sc))

silh = silhouette_score1(data_nor,  result_sc)
u,indices = np.unique(result_sc,return_counts = True) # sc=0.3412 st 64
# u,indices
print(silh)
print(u,indices)

from sklearn.metrics import calinski_harabasz_score
ch_index = calinski_harabasz_score(data_nor, result_sc)

print(ch_index)

print("RMSE score is ", total_rmse('/content/drive/MyDrive/Data/ERA5_meteo_sfc_2021_daily.nc', result_sc))

print("Variance is ", avg_var(data_nor, result_sc))

print("Inter-cluster distance ", avg_inter_dist(data_nor, result_sc))

