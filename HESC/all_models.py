# -*- coding: utf-8 -*-
"""all_models.ipynb

"""

from google.colab import drive
drive.mount('/content/drive')

"""#Install needed libraries"""

# Install dask.dataframe
!pip install "dask[dataframe]"
!pip install netCDF4
!pip install PyMetis
!pip install kahypar

import os
import warnings
from typing import Optional
import numpy as np
import pandas as pd
import xarray as xr
import pymetis
import kahypar
from scipy import sparse
from sklearn.metrics import pairwise_distances, normalized_mutual_info_score
from sklearn.utils.extmath import safe_sparse_dot

from sklearn.metrics import silhouette_score, pairwise_distances, davies_bouldin_score
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import random
import netCDF4 as nc
import datetime
import datetime as dt
from netCDF4 import date2num,num2date
from math import sqrt
from time import time
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
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import sys
import pickle
import matplotlib as mpl
import matplotlib.colors as colors

warnings.filterwarnings("ignore")


def silhouette_score1(X, labels, *, metric="cosine", sample_size=None, random_state=None, **kwds):
 return np.mean(silhouette_samples(X, labels, metric="cosine", **kwds))

## This function will will pre-process our daily data for DEC model as numpy array
from sklearn import preprocessing

warnings.filterwarnings("ignore")

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

path2 = ('/content/drive/MyDrive/Data/ERA5_meteo_sfc_2021_daily.nc')
data = xr.open_dataset(path2, decode_times=False)#To view the date as integers of 0, 1, 2,....
var = list(data.variables)[3:]

path = '/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/'

data_nor, data_clustering = data_preprocessing('/content/drive/MyDrive/Data/ERA5_meteo_sfc_2021_daily.nc')

data_nor.shape, data_clustering.shape

data_nor_nor = data_nor

###################################################################

##DSC homogeneous clustering results
DSC_cluster_result = pickle.load(open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/DSC_Ensemble1_032.pkl", "rb"))

DSC_cluster_result1 = pickle.load(open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/DEC_hom_ens_0.34805576652439557.pkl", "rb"))

DSC_cluster_result2 = pickle.load(open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/DSC_hom_ens_0.35190532062602214.pkl", "rb"))

###################################################################

##DEC homogeneous clustering results
DEC_Cluster_results1 = pickle.load(open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/DEC_Ensemble1_034.pkl", "rb"))

DEC_Cluster_results2 = pickle.load(open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/DEC_hom_ens_0.30096328.pkl", "rb"))

DEC_Cluster_results3 = pickle.load(open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/DEC_hom_ens_0.30640745.pkl", "rb"))

DEC_Cluster_results4 = pickle.load(open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/DEC_hom_ens_03315.pkl", "rb"))

###################################################################

##DTC homogeneous clustering results
DTC_Cluster_results1 = pickle.load(open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/DEC_Ensemble1_034.pkl", "rb"))

DTC_Cluster_results2 = pickle.load(open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/DTC_3367.pkl", "rb"))

DTC_Cluster_results3 = pickle.load(open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/DTC_hom_ens_034.pkl", "rb"))

DTC_Cluster_results4 = pickle.load(open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/DTC_3226.pkl", "rb"))

DTC_Cluster_results5 = pickle.load(open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/DTC_3124.pkl", "rb"))

###################################################################

##KMeans homogeneous clustering results
KMeans_Cluster_results1 = pickle.load(open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/KMeans_0.3389.pkl", "rb"))

KMeans_Cluster_results2 = pickle.load(open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/KMeans_0.313.pkl", "rb"))

KMeans_Cluster_results3 = pickle.load(open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/KMeans_0.3125.pkl", "rb"))

###################################################################

##Main Heterogeneous Ensemble
"""# **HESC_performance**"""

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

NUM_alg = 10
occurrence_threshold = 0.2

clustering_models = NUM_alg * [KMeans_Cluster_results1, DTC_Cluster_results1, DTC_Cluster_results2,
DEC_Cluster_results1, DEC_Cluster_results2, DEC_Cluster_results3, DEC_Cluster_results4, NUM_alg * DSC_cluster_result, NUM_alg * DSC_cluster_result1, NUM_alg * DSC_cluster_result2]


# KMeans_Cluster_results1, KMeans_Cluster_results2, KMeans_Cluster_results3, DTC_Cluster_results1, DTC_Cluster_results2, DTC_Cluster_results3, DTC_Cluster_results4,
# DEC_Cluster_results1, DEC_Cluster_results4, DSC_cluster_result, DSC_cluster_result1

clt_sim_matrix = ClusterSimilarityMatrix() #Initializing the similarity matrix
for model in clustering_models:
  clt_sim_matrix.fit(model)

sim_matrix = clt_sim_matrix.similarity
sim_matrixx = sim_matrix/sim_matrix.diagonal()
sim_matrixx[sim_matrixx < occurrence_threshold] = 0

# norm_sim_matrix[norm_sim_matrix < occurrence_threshold] = 0

unique_labels = np.unique(np.concatenate(sim_matrixx))

print(sim_matrixx)
print(sim_matrixx.shape)
print(unique_labels)
#np.save('/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Non-negative Matrix Factorization/fin_ens_co_occurrence_matrix.npy', sim_matrixx)
#print(norm_sim_matrix)

# To normalize a matrix
# dfmax, dfmin = df.max(), df.min()

# df = (df - dfmin)/(dfmax - dfmin)

# print(df)

# def build_binary_matrix(clabels ):

#   data_len = len(clabels)

#   matrix=np.zeros((data_len,data_len))
#   for i in range(data_len):
#     matrix[i,:] = clabels == clabels[i]
#   return matrix

# build_binary_matrix(KMeans_Cluster_results1)

from sklearn.cluster import SpectralClustering
spec_clt = SpectralClustering(n_clusters=7, affinity='precomputed',
                              n_init=5, random_state=214)
final_labels = spec_clt.fit_predict(sim_matrixx)

# final_labels

silh = silhouette_score1(data_nor,  final_labels)
u,indices = np.unique(final_labels,return_counts = True) # sc=0.3412 st 64
# u,indices
print(silh)
print(u,indices)

#pickle.dump(final_labels, open(path + 'co-occ_ens_' + str(silh) + '.pkl', "wb"))

"""**Davies Bouldin**"""

from sklearn.metrics import davies_bouldin_score

print("Davies-Bouldin score is ", davies_bouldin_score(data_nor, final_labels))



total_rmse('/content/drive/MyDrive/Data/ERA5_meteo_sfc_2021_daily.nc', final_labels)

avg_var(data_nor, final_labels)

avg_inter_dist(data_nor, final_labels)



"""# **Co-Occurrence matrix**"""

from sklearn.metrics import pairwise_distances

dsc_ens = pickle.load(open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/DSC_fin_ens_0.32257442534935293.pkl", "rb"))
dec_ens = pickle.load(open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/DEC_fin_ens_0.3135124.pkl", "rb"))
km_ens = pickle.load(open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/kmeans_fin_ens_0.32412578566580286.pkl", "rb"))

#dtc_ens = pickle.load(open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/DEC_fin_ens_0.3135124.pkl", "rb"))



NUM_alg = 10
occurrence_threshold = 0.0

clustering_models = NUM_alg * [dsc_ens, dec_ens, km_ens]

# clustering_models = NUM_alg * [KMeans_Cluster_results1, KMeans_Cluster_results2, KMeans_Cluster_results3, DTC_Cluster_results1, DTC_Cluster_results2, DTC_Cluster_results3, DTC_Cluster_results4,
# DEC_Cluster_results1, DEC_Cluster_results2, DEC_Cluster_results3, DEC_Cluster_results4, DSC_cluster_result, DSC_cluster_result1]


# KMeans_Cluster_results1, KMeans_Cluster_results2, KMeans_Cluster_results3, DTC_Cluster_results1, DTC_Cluster_results2, DTC_Cluster_results3, DTC_Cluster_results4,
# DEC_Cluster_results1, DEC_Cluster_results2, DEC_Cluster_results3, DEC_Cluster_results4, DSC_cluster_result, DSC_cluster_result1

clt_sim_matrix = ClusterSimilarityMatrix() #Initializing the similarity matrix
for model in clustering_models:
  clt_sim_matrix.fit(model)

sim_matrix = clt_sim_matrix.similarity
sim_matrixx = sim_matrix/sim_matrix.diagonal()
sim_matrixx[sim_matrixx < occurrence_threshold] = 0

# norm_sim_matrix[norm_sim_matrix < occurrence_threshold] = 0

unique_labels = np.unique(np.concatenate(sim_matrixx))

print(sim_matrixx)
print(sim_matrixx.shape)
print(unique_labels)
np.save('/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Non-negative Matrix Factorization/fin_ens_co_occurrence_matrix.npy', sim_matrixx)
#print(norm_sim_matrix)



from sklearn.cluster import SpectralClustering
spec_clt = SpectralClustering(n_clusters=7, affinity='precomputed',
                              n_init=5, random_state=214)
final_labels = spec_clt.fit_predict(sim_matrixx)

final_labels

silh = silhouette_score1(data_nor,  final_labels)
u,indices = np.unique(final_labels,return_counts = True) # sc=0.3412 st 64
# u,indices
print(silh)
print(u,indices)

pickle.dump(final_labels, open(path + 'best_co-occ_ens_' + str(silh) + '.pkl', "wb"))

"""**Davies Bouldin**"""

from sklearn.metrics import davies_bouldin_score

print("Davies-Bouldin score is ", davies_bouldin_score(data_nor, final_labels))

from sklearn.metrics import calinski_harabasz_score
ch = calinski_harabasz_score(data_nor, final_labels)
print("Davies-Bouldin score is ", ch)

total_rmse('/content/drive/MyDrive/Data/ERA5_meteo_sfc_2021_daily.nc', final_labels)

avg_var(data_nor, final_labels)

avg_inter_dist(data_nor, final_labels)



"""# **Non-negative Matrix Factorization**"""

from sklearn.decomposition import NMF

sim_matrixx = np.load('/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Non-negative Matrix Factorization/fin_ens_co_occurrence_matrix.npy')

#sim_matrixx

def best_clustering5(n):

  n # Number of single models used
    #MIN_PROBABILITY = 0.6 # The minimum threshold of occurances of datapoints in a cluster

  sil = []
  sil_score = []
  # Generating a "Cluster Forest"
  #clustering_models = NUM_alg * [main()]#
  for i in range(n):
    r = 7

    # Create an NMF model with the specified number of components
    # model = NMF(n_components=5, random_state=1, alpha=.1, l1_ratio=0.5)
    model = NMF(n_components=r, init='random', random_state=0)

    # Fit the model to the data
    W = model.fit_transform(sim_matrixx)
    H = model.components_

    # Reconstruct the data matrix
    X_approximated = np.dot(W, H)

    clusters_nmf = np.argmax(W, axis = 1)

    silhouette_avg_rdata_daily = silhouette_score1(data_nor_eval, clusters_nmf)

    sil_score.append(silhouette_avg_rdata_daily)

    sil.append([silhouette_avg_rdata_daily, clusters_nmf])

  print("Our silhouettes range is: ", sil_score)

  max_index = 0

  for j in range(len(sil)):
    if(sil[j][0]>=sil[max_index][0]):
      max_index = j
  print(sil[max_index])

  #return sil[max_index]
  return sil[max_index]

silhouette, result_nmf = best_clustering5(20)

silh = silhouette_score1(data_nor,  result_nmf)
u,indices = np.unique(result_nmf,return_counts = True) # sc=0.3412 st 64
# u,indices
print(silh)
print(u,indices)

# Specify the number of components (r)
r = 7

# Create an NMF model with the specified number of components
# model = NMF(n_components=5, random_state=1, alpha=.1, l1_ratio=0.5)
model = NMF(n_components=r, init='random', random_state=0)

# Fit the model to the data
W = model.fit_transform(sim_matrixx)
H = model.components_

# Reconstruct the data matrix
X_approximated = np.dot(W, H)


print("Original Data Matrix (X):\n", sim_matrixx)
print("\nBasis Matrix (W):\n", W)
print("\nCoefficient Matrix (H):\n", H)
print("\nApproximated Data Matrix (X_approximated):\n", X_approximated)

clusters_nmf = np.argmax(W, axis = 1)

clusters_nmf



silh = silhouette_score1(data_nor,  clusters_nmf)
u,indices = np.unique(clusters_nmf,return_counts = True) # sc=0.3412 st 64
# u,indices
print(silh)
print(u,indices)

pickle.dump(clusters_nmf, open(path + 'co-occ_ens_' + str(silh) + '.pkl', "wb"))

"""**Davies Bouldin**"""

from sklearn.metrics import davies_bouldin_score

print("Davies-Bouldin score is ", davies_bouldin_score(data_nor, clusters_nmf))

total_rmse('/content/drive/MyDrive/Data/ERA5_meteo_sfc_2021_daily.nc', clusters_nmf)

avg_var(data_nor, clusters_nmf)

avg_inter_dist(data_nor, clusters_nmf)

"""# **Hybrid Bipartite Graph Formulation (HBGF)**"""

def create_hypergraph(base_clusters):
    """Create the incidence matrix of base clusters' hypergraph

    Parameter
    ----------
    base_clusters: labels generated by base clustering algorithms

    Return
    -------
    H: incidence matrix of base clusters' hypergraph
    """
    H = []
    len_bcs = base_clusters.shape[1]

    for bc in base_clusters:
        bc = np.nan_to_num(bc, nan=float('inf'))
        unique_bc = np.unique(bc)
        len_unique_bc = len(unique_bc)
        bc2id = dict(zip(unique_bc, np.arange(len_unique_bc)))
        tmp = [bc2id[bc_elem] for bc_elem in bc]
        h = np.identity(len_unique_bc, dtype=int)[tmp]
        if float('inf') in bc2id.keys():
            h = np.delete(h, obj=bc2id[float('inf')], axis=1)
        H.append(sparse.csc_matrix(h))

    return sparse.hstack(H)

def to_pymetis_format(adj_mat):
    """Transform an adjacency matrix into the pymetis format

    Parameter
    ---------
    adj_mat: adjacency matrix

    Returns
    -------
    xadj, adjncy, eweights: parameters for pymetis
    """
    xadj = [0]
    adjncy = []
    eweights = []
    n_rows = adj_mat.shape[0]
    adj_mat = adj_mat.tolil()

    for i in range(n_rows):
        row = adj_mat.getrow(i)
        idx_row, idx_col = row.nonzero()
        val = row[idx_row, idx_col]
        adjncy += list(idx_col)
        # eweights += list(val.toarray()[0])
        eweights += list(val[0])
        xadj.append(len(adjncy))

    return xadj, adjncy, eweights

def hbgf(base_clusters, nclass):
    """Hybrid Bipartite Graph Formulation (HBGF)

    Parameters
    ----------
    base_clusters: labels generated by base clustering algorithms
    nclass: number of classes

    Return
    -------
    celabel: consensus clustering label obtained from HBGF
    """
    A = create_hypergraph(base_clusters)
    rowA, colA = A.shape
    W = sparse.bmat([[sparse.dok_matrix((colA, colA)), A.T],
                    [A, sparse.dok_matrix((rowA, rowA))]])
    xadj, adjncy, _ = to_pymetis_format(W)
    membership = pymetis.part_graph(
        nparts=nclass, xadj=xadj, adjncy=adjncy, eweights=None)[1]
    celabel = np.array(membership[colA:])
    return celabel

label_hbgf = hbgf(np.array(clustering_models), 7)

# labe = hbgf(sim_matrixx, 7)

label_hbgf

silh = silhouette_score1(data_nor_eval,  label_hbgf)
u,indices = np.unique(label_hbgf,return_counts = True) # sc=0.3412 st 64
# u,indices
print(silh)
print(u,indices)

#path = '/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/'
pickle.dump(label_hbgf, open(path + 'HBGF_ens_' + str(silh) + '.pkl', "wb"))

silh = silhouette_score1(data_nor,  label_hbgf)
u,indices = np.unique(label_hbgf,return_counts = True) # sc=0.3412 st 64
# u,indices
print(silh)
print(u,indices)

print("Davies-Bouldin score is ", davies_bouldin_score(data_nor, label_hbgf))

print("Calinski Harabas score is ", calinski_harabasz_score(data_nor, label_hbgf))

print("RMSE score is ", total_rmse('/content/drive/MyDrive/Data/ERA5_meteo_sfc_2021_daily.nc', label_hbgf))

print("Variance is ", avg_var(data_nor, label_hbgf))

print("Inter-cluster distance ", avg_inter_dist(data_nor, label_hbgf))

def best_clustering(n):

  n # Number of single models used
    #MIN_PROBABILITY = 0.6 # The minimum threshold of occurances of datapoints in a cluster

  sil = []
  sil_score = []

  for i in range(n):
    results = hbgf(np.array(clustering_models), 7)
    #sil.append(results)
    #print(results)
    silhouette_avg_rdata_daily = silhouette_score1(data_nor, results)
    sil_score.append(silhouette_avg_rdata_daily)

    sil.append([silhouette_avg_rdata_daily, results])

  print("Our silhouettes range is: ", sil_score)

  max_index = 0

  for j in range(len(sil)):
    if(sil[j][0]>=sil[max_index][0]):
      max_index = j
  print(sil[max_index])

  return sil[max_index]

sil, resultss = best_clustering(20)

pickle.dump(resultss, open(path + 'HBGF_enss_' + str(silh) + '.pkl', "wb"))

silh = silhouette_score1(data_nor_eval,  resultss)
u,indices = np.unique(resultss,return_counts = True) # sc=0.3412 st 64
# u,indices
print(silh)
print(u,indices)

print("Davies-Bouldin score is ", davies_bouldin_score(data_nor, resultss))

print("Calinski Harabas score is ", calinski_harabasz_score(data_nor, resultss))

print("RMSE score is ", total_rmse('/content/drive/MyDrive/Data/ERA5_meteo_sfc_2021_daily.nc', resultss))

print("Variance is ", avg_var(data_nor, resultss))

print("Inter-cluster distance ", avg_inter_dist(data_nor, resultss))



"""# **HESC_hbgp**"""

hbgp = pickle.load(open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/HBGF_enss_0.30932982840030593.pkl", "rb"))

silh = silhouette_score1(data_nor_eval,  hbgp)
u,indices = np.unique(hbgp,return_counts = True) # sc=0.3412 st 64
# u,indices
print(silh)
print(u,indices)

from sklearn.metrics import davies_bouldin_score

print("Davies-Bouldin score is ", davies_bouldin_score(data_nor, final_labels))

total_rmse('/content/drive/MyDrive/Data/ERA5_meteo_sfc_2021_daily.nc', hbgp)

avg_var(data_nor, hbgp)

avg_inter_dist(data_nor, hbgp)

"""# **Final Consensus**"""



co_occ = pickle.load(open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/co-occ_ens_0.35889223651046626.pkl", "rb"))

nnmf = pickle.load(open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/nmf_ens0.35258519039733466.pkl", "rb"))

NUM_alg = 100
occurrence_threshold = 0.2

final_models = NUM_alg  * [co_occ, nnmf]


clt_sim_matrix = ClusterSimilarityMatrix() #Initializing the similarity matrix
for model in final_models:
  clt_sim_matrix.fit(model)

final_matrix = clt_sim_matrix.similarity
final_matrix = final_matrix/final_matrix.diagonal()
final_matrix[final_matrix < occurrence_threshold] = 0

unique_labels = np.unique(np.concatenate(sim_matrixx))

print(final_matrix)
print(final_matrix.shape)
print(unique_labels)

#final_labels1 = pickle.load(open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/co-occ_ens_0.35936684974088645.pkl", "rb"))
ch_index1s2 = calinski_harabasz_score(data_nor, nnmf)

print(ch_index1s2)

from sklearn.cluster import SpectralClustering
spec_clt = SpectralClustering(n_clusters=7, affinity='precomputed',
                              n_init=5, random_state=214)
final_labels = spec_clt.fit_predict(final_matrix)

final_labels

silh = silhouette_score1(data_nor,  final_labels)
u,indices = np.unique(final_labels,return_counts = True) # sc=0.3412 st 64
# u,indices
print(silh)
print(u,indices)



print("Davies-Bouldin score is ", davies_bouldin_score(data_nor, final_labels))

print("Calinski Harabas score is ", calinski_harabasz_score(data_nor, final_labels))

print("RMSE score is ", total_rmse('/content/drive/MyDrive/Data/ERA5_meteo_sfc_2021_daily.nc', final_labels))

print("Variance is ", avg_var(data_nor, final_labels))

print("Inter-cluster distance ", avg_inter_dist(data_nor, final_labels))

"""**Davies Bouldin**"""

from sklearn.metrics import davies_bouldin_score

print("Davies-Bouldin score is ", davies_bouldin_score(data_nor, final_labels))

"""# **HESC**"""

from sklearn.metrics import calinski_harabasz_score

from sklearn.metrics import calinski_harabasz_score
ch_index = calinski_harabasz_score(data_nor, final_labels)

print(ch_index)



"""# **KMeans**"""

from sklearn.metrics import calinski_harabasz_score



final_labels1 = pickle.load(open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/co-occ_ens_0.35936684974088645.pkl", "rb"))
ch_index12 = calinski_harabasz_score(data_nor, final_labels1)

print(ch_index12)

"""# **DTC**"""

final_labels2 = pickle.load(open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/DTC_22.pkl", "rb"))

ch_index2 = calinski_harabasz_score(data_nor, final_labels2)

print(ch_index2)

"""# **DSC**"""

final_labels3 = pickle.load(open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/DSC_hom_ens_0.35190532062602214.pkl", "rb"))
#/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/DSC_hom_ens_0.32763868041010935.pkl
ch_index3 = calinski_harabasz_score(data_nor, final_labels3)

print(ch_index3)

"""# **DEC**"""

final_labels4 = pickle.load(open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/DEC_hom_ens_03315.pkl", "rb"))

ch_index4 = calinski_harabasz_score(data_nor, final_labels4)

print(ch_index4)
