



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

!pip install torch-geometric

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

import networkx as nx
import torch
import csv
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import matplotlib.pyplot as plt

"""# Importing Datasets"""

#path2 = ('/content/drive/MyDrive/Data/mock_v4.nc')
path2 = ('/content/multivariate-weather-data-clustering/data/ERA5_meteo_sfc_2021_daily.nc')
#path2 = ('/content/drive/MyDrive/Data/ERA5_meteo_sfc_2021_daily.nc')
#path2 = ('/content/drive/MyDrive/Data/ERA5_meteo_sfc_2021_daily_smalldomain.nc')
#path2 = ('/content/drive/MyDrive/Data/ERA5_meteo_sfc_2021_hourly.nc')
#path2 = ('/content/drive/MyDrive/Data/ERA5_meteo_sfc_2021_hourly_smalldomain.nc')
data = xr.open_dataset(path2, decode_times=False)#To view the date as integers of 0, 1, 2,....
#data = xr.open_dataset(path2)# decode_times=False) #To view the date as integers of 0, 1, 2,....
#data5 = xr.open_dataset(path2) # To view time in datetime format
var = list(data.variables)[3:]
data

def datatransformation(data_path, variables):
  ''' The parameters accepted by this function are as follows:
    1. "data_path" is the path of the netCDF4 dataset file. (data_path = '/content/drive/MyDrive/ERA5_meteo_sfc_2021_daily.nc')
    2. "variables" is an array of the variable names of the netCDF4 dataset those we want to read. (variables = ['sst', 'sp'])
        If the "variables" array is empty the function will read the whole dataset.

    Return value:
       The function will return the normalized values of the selected variables as a 2D NumPy array of size (365 x ___)  and a 4D array as (365, 41, 41, ___).
      '''

  rdata_daily = xr.open_dataset(data_path)    # data_path = '/content/drive/MyDrive/ERA5_Dataset.nc'
  if(len(variables)==0):
    rdata_daily_np_array = np.array(rdata_daily.to_array()) # the shape of the dailt data is (7, 365, 41, 41)
  else:
    rdata_daily_np_array = np.array(rdata_daily[variables].to_array())
  rdata_daily_np_array_R = rdata_daily_np_array.reshape((rdata_daily_np_array.shape[0], -1)) #(7, 613565)
  for i in range (rdata_daily_np_array_R.shape[0]):
    tmp = rdata_daily_np_array_R[i]
    tmp[np.isnan(tmp)]=np.nanmean(tmp)
    rdata_daily_np_array_R[i] = tmp
  min_max_scaler = MinMaxScaler() # calling the function
  rdata_daily_np_array_nor =  min_max_scaler.fit_transform(rdata_daily_np_array_R.T).T
  rdata_daily_np_array_nor_4D = rdata_daily_np_array_nor.reshape(rdata_daily_np_array.shape) # (7, 613565) to (7, 365, 41, 41)
  rdata_daily_np_array_nor_4D_T = rdata_daily_np_array_nor_4D.transpose(1,2,3,0)   #  (7, 365, 41, 41) to (365, 41, 41, 7)
  rdata_daily_np_array_nor_4D_T_R = rdata_daily_np_array_nor_4D_T.reshape((rdata_daily_np_array_nor_4D_T.shape[0], -1)) #(365, 11767)
  data_2d = rdata_daily_np_array_nor_4D_T_R
  data_4d = rdata_daily_np_array_nor_4D_T
  return data_2d



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


def cspa(base_clusters, nclass):
    """Cluster-based Similarity Partitioning Algorithm (CSPA)

    Parameters
    ----------
    base_clusters: labels generated by base clustering algorithms
    nclass: number of classes

    Return
    -------
    celabel: consensus clustering label obtained from CSPA
    """
    H = create_hypergraph(base_clusters)
    S = H * H.T

    xadj, adjncy, eweights = to_pymetis_format(S)
    membership = pymetis.part_graph(
        nparts=nclass, xadj=xadj, adjncy=adjncy, eweights=eweights)[1]
    celabel = np.array(membership)

    return celabel


def hgpa(base_clusters, nclass, random_state):
    """HyperGraph Partitioning Algorithm (HGPA)

    Parameters
    ----------
    base_clusters: labels generated by base clustering algorithms
    nclass: number of classes
    random_state: used for reproducible results

    Return
    -------
    celabel: consensus clustering label obtained from HGPA
    """
    # Create hypergraph for kahypar
    H = create_hypergraph(base_clusters)
    n_nodes, n_nets = H.shape

    node_weights = [1] * n_nodes
    edge_weights = [1] * n_nets

    hyperedge_indices = [0]
    hyperedges = []
    HT = H.T
    for i in range(n_nets):
        h = HT.getrow(i)
        idx_row, idx_col = h.nonzero()
        hyperedges += list(idx_col)
        hyperedge_indices.append(len(hyperedges))

    hypergraph = kahypar.Hypergraph(
        n_nodes, n_nets, hyperedge_indices, hyperedges, nclass, edge_weights, node_weights)

    # Settings for kahypar
    context = kahypar.Context()
    config_path = os.path.dirname(
        __file__) + '/kahypar_config/km1_kKaHyPar_sea20.ini'
    context.loadINIconfiguration(config_path)
    if random_state is not None:
        context.setSeed(random_state)
    context.setK(nclass)
    context.setEpsilon(0.03)
    context.suppressOutput(True)

    # Hypergraph partitioning
    kahypar.partition(hypergraph, context)

    celabel = np.empty(n_nodes, dtype=int)
    for i in range(n_nodes):
        celabel[i] = hypergraph.blockID(i)

    return celabel


def mcla(base_clusters, nclass, random_state):
    """Meta-CLustering Algorithm (MCLA)

    Parameters
    ----------
    base_clusters: labels generated by base clustering algorithms
    nclass: number of classes
    random_state: used for reproducible results

    Return
    -------
    celabel: consensus clustering label obtained from MCLA
    """
    np.random.seed(random_state)

    # Construct Meta-graph
    H = create_hypergraph(base_clusters)
    n_cols = H.shape[1]

    W = sparse.identity(n_cols, dtype=float, format='lil')
    for i in range(n_cols):
        hi = H.getcol(i)
        norm_hi = (hi.T * hi)[0, 0]
        for j in range(n_cols):
            if i < j:
                hj = H.getcol(j)
                norm_hj = (hj.T * hj)[0, 0]
                inner_prod = (hi.T * hj)[0, 0]
                W[i, j] = inner_prod / (norm_hi + norm_hj - inner_prod)
                W[j, i] = W[i, j]
    W *= 1e3
    W = W.astype(int)

    # Cluster Hyperedges
    xadj, adjncy, eweights = to_pymetis_format(W)
    membership = pymetis.part_graph(
        nparts=nclass, xadj=xadj, adjncy=adjncy, eweights=eweights)[1]

    # Collapse Meta-clusters
    meta_clusters = sparse.dok_matrix(
        (base_clusters.shape[1], nclass), dtype=float).tolil()
    for i, v in enumerate(membership):
        meta_clusters[:, v] += H.getcol(i)

    # Compete for Objects
    celabel = np.empty(base_clusters.shape[1], dtype=int)
    for i, v in enumerate(meta_clusters):
        v = v.toarray()[0]
        celabel[i] = np.random.choice(np.nonzero(v == np.max(v))[0])

    return celabel


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


def create_connectivity_matrix(base_clusters):
    """Create the connectivity matrix

    Parameter
    ---------
    base_clusters: labels generated by base clustering algorithms

    Return
    ------
    M: connectivity matrix
    """
    n_bcs, len_bcs = base_clusters.shape
    M = np.zeros((len_bcs, len_bcs))
    m = np.zeros_like(M)

    for bc in base_clusters:
        for i, elem_bc in enumerate(bc):
            m[i] = np.where(elem_bc == bc, 1, 0)
        M += m

    M /= n_bcs
    return sparse.csr_matrix(M)


def orthogonal_nmf_algorithm(W, nclass, random_state, maxiter):
    """Algorithm for bi-orthogonal three-factor NMF problem

    Parameters
    ----------
    W: given matrix
    random_state: used for reproducible results
    maxiter: maximum number of iterations

    Return
    -------
    Q, S: factor matrices
    """
    np.random.seed(random_state)

    n = W.shape[0]
    Q = np.random.rand(n, nclass).reshape(n, nclass)
    S = np.diag(np.random.rand(nclass))

    for _ in range(maxiter):
        # Update Q
        WQS = safe_sparse_dot(W, np.dot(Q, S), dense_output=True)
        Q = Q * np.sqrt(WQS / (np.dot(Q, np.dot(Q.T, WQS)) + 1e-8))
        # Update S
        QTQ = np.dot(Q.T, Q)
        WQ = safe_sparse_dot(W, Q, dense_output=False)
        QTWQ = safe_sparse_dot(Q.T, WQ, dense_output=True)
        S = S * np.sqrt(QTWQ / (np.dot(QTQ, np.dot(S, QTQ)) + 1e-8))

    return Q, S


def nmf(base_clusters, nclass, random_state, maxiter=200):
    """NMF-based consensus clustering

    Parameters
    ----------
    base_clusters: labels generated by base clustering algorithms
    nclass: number of classes
    random_state: used for reproducible results
    maxiter: maximum number of iterations

    Return
    -------
    celabel: consensus clustering label obtained from NMF
    """
    M = create_connectivity_matrix(base_clusters)
    Q, S = orthogonal_nmf_algorithm(M, nclass, random_state, maxiter)
    celabel = np.argmax(np.dot(Q, np.sqrt(S)), axis=1)
    return celabel


def calc_objective(base_clusters, consensus_cluster):
    """Calculate the objective function value for cluster ensembles

    Parameters
    ----------
    base_clusters: labels generated by base clustering algorithms
    consensus_cluster: consensus clustering label

    Return
    -------
    objv: objective function value
    """
    objv = 0.0
    for bc in base_clusters:
        idx = np.isfinite(bc)
        objv += normalized_mutual_info_score(
            consensus_cluster[idx], bc[idx], average_method='geometric')
    objv /= base_clusters.shape[0]
    return objv


def cluster_ensembles(
        base_clusters: np.ndarray,
        nclass: Optional[int] = None,
        solver: str = 'nmf',
        random_state: Optional[int] = None,
        verbose: bool = False) -> np.ndarray:
    """Generate a single consensus cluster using base clusters obtained from multiple clustering algorithms

    Parameters
    ----------
    base_clusters: labels generated by base clustering algorithms
    nclass: number of classes
    solver: cluster ensembles solver to use
    random_state: used for 'hgpa', 'mcla', and 'nmf'. Pass a nonnegative integer for reproducible results.
    verbose: whether to be verbose

    Return
    -------
    celabel: consensus clustering label
    """
    if nclass is None:
        nclass = -1
        for bc in base_clusters:
            len_unique_bc = len(np.unique(bc[~np.isnan(bc)]))
            nclass = max(nclass, len_unique_bc)

    if verbose:
        print('Cluster Ensembles')
        print('    - number of classes:', nclass)
        print('    - solver:', solver)
        print('    - length of base clustering labels:',
              base_clusters.shape[1])
        print('    - number of base clusters:', base_clusters.shape[0])

    if not (isinstance(nclass, int) and nclass > 0):
        raise ValueError(
            'Number of class must be a positive integer; got (nclass={})'.format(nclass))

    if not ((random_state is None) or isinstance(random_state, int)):
        raise ValueError(
            'Number of random_state must be a nonnegative integer; got (random_state={})'.format(random_state))

    if isinstance(random_state, int):
        random_state = abs(random_state)

    if solver == 'cspa':
        if base_clusters.shape[1] > 5000:
            warnings.warn(
                '`base_clusters.shape[1]` is too large, so the use of another solvers is recommended.')
        celabel = cspa(base_clusters, nclass)
    elif solver == 'hgpa':
        celabel = hgpa(base_clusters, nclass, random_state)
    elif solver == 'mcla':
        celabel = mcla(base_clusters, nclass, random_state)
    elif solver == 'hbgf':
        celabel = hbgf(base_clusters, nclass)
    elif solver == 'nmf':
        celabel = nmf(base_clusters, nclass, random_state)
    elif solver == 'all':
        if verbose:
            print('    - ANMI:')
        ce_solvers = {'hgpa': hgpa, 'mcla': mcla, 'hbgf': hbgf}
        if base_clusters.shape[1] <= 5000:
            ce_solvers['cspa'] = cspa
            ce_solvers['nmf'] = nmf
        best_objv = None
        for name, ce_solver in ce_solvers.items():
            if ce_solver == cspa or ce_solver == hbgf:
                label = ce_solver(base_clusters, nclass)
            else:
                label = ce_solver(base_clusters, nclass, random_state)
            objv = calc_objective(base_clusters, label)
            if verbose:
                print('        -', name, ':', objv)
            if best_objv is None:
                best_objv = objv
                best_solver = name
                celabel = label
            if best_objv < objv:
                best_objv = objv
                best_solver = name
                celabel = label
        if verbose:
            print('    - Best solver:', best_solver)
    else:
        raise ValueError(
            "Invalid solver parameter: got '{}' instead of one of ('cspa', 'hgpa', 'mcla', 'hbgf', 'nmf', 'all')".format(solver))

    return celabel

df = datatransformation(path2, var)

def silhouette_score1(X, labels, *, metric="cosine", sample_size=None, random_state=None, **kwds):
 return np.mean(silhouette_samples(X, labels, metric="cosine", **kwds))

kmeans = KMeans(n_clusters=7, random_state=1, n_init="auto")
kmeans.fit(df)
P1 = kmeans.predict(df)
labels1 =  kmeans.labels_
labels1

print("Silhouette Coefficient score is ", silhouette_score1(df , labels1))

kmeans = KMeans(n_clusters=7, random_state=2, n_init="auto")
kmeans.fit(df)
P2 = kmeans.predict(df)
labels2 =  kmeans.labels_
labels2

print("Silhouette Coefficient score is ", silhouette_score1(df , labels2))

kmeans = KMeans(n_clusters=7, random_state=3, n_init="auto")
kmeans.fit(df)
P3 = kmeans.predict(df)
labels3 =  kmeans.labels_
labels3

print("Silhouette Coefficient score is ", silhouette_score1(df , labels3))

kmeans = KMeans(n_clusters=7, random_state=4, n_init="auto")
kmeans.fit(df)
P4 = kmeans.predict(df)
labels4 =  kmeans.labels_
labels4

print("Silhouette Coefficient score is ", silhouette_score1(df , labels4))

kmeans = KMeans(n_clusters=7, random_state=5, n_init="auto")
kmeans.fit(df)
P5 = kmeans.predict(df)
labels5 =  kmeans.labels_
labels5

print("Silhouette Coefficient score is ", silhouette_score1(df , labels5))

kmeans = KMeans(n_clusters=7, random_state=6, n_init="auto")
kmeans.fit(df)
P6 = kmeans.predict(df)
labels6 =  kmeans.labels_
labels6

print("Silhouette Coefficient score is ", silhouette_score1(df , labels6))

kmeans = KMeans(n_clusters=7, random_state=7, n_init="auto")
kmeans.fit(df)
P7 = kmeans.predict(df)
labels7 =  kmeans.labels_
labels7

print("Silhouette Coefficient score is ", silhouette_score1(df , labels7))

kmeans = KMeans(n_clusters=7, random_state=8, n_init="auto")
kmeans.fit(df)
P8 = kmeans.predict(df)
labels8 =  kmeans.labels_
labels8

print("Silhouette Coefficient score is ", silhouette_score1(df , labels8))

kmeans = KMeans(n_clusters=7, random_state=9, n_init="auto")
kmeans.fit(df)
P9 = kmeans.predict(df)
labels9 =  kmeans.labels_
labels9

print("Silhouette Coefficient score is ", silhouette_score1(df , labels9))

kmeans = KMeans(n_clusters=7, random_state=10, n_init="auto")
kmeans.fit(df)
P10 = kmeans.predict(df)
labels10 =  kmeans.labels_
labels10

print("Silhouette Coefficient score is ", silhouette_score1(df , labels10))

labels10.shape

labels = np.array([labels1, labels2, labels3, labels4, labels5, labels6, labels7, labels8, labels9, labels10])
labels.shape

label_CE = cluster_ensembles(labels)
label_CE.shape

label_CE = cluster_ensembles(labels, 7,'nmf', 150, False) #'hgpa', 'mcla', and 'nmf'

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

silh = silhouette_score1(data_nor,  label_CE)
u,indices = np.unique(label_CE,return_counts = True)

print("Silhouette score is ", silh)
print("Cluster index ", u, "and Cluster Sizes: ", indices)

from sklearn.metrics import davies_bouldin_score

db = davies_bouldin_score(data_nor, label_CE)
print("Davies-Bouldin score is ", db)

from sklearn.metrics import calinski_harabasz_score
ch = calinski_harabasz_score(data_nor, label_CE)
print("Davies-Bouldin score is ", ch)

print("RMSE score is ", total_rmse('/content/drive/MyDrive/Data/ERA5_meteo_sfc_2021_daily.nc', label_CE))

print("Variance is ", avg_var(data_nor, label_CE))

print("Inter-cluster distance ", avg_inter_dist(data_nor, label_CE))



"""#Loop multiple times"""

def best_clustering(n):

  n # Number of single models used
    #MIN_PROBABILITY = 0.6 # The minimum threshold of occurances of datapoints in a cluster

  sil = []
  sil_score = []
  # Generating a "Cluster Forest"
  #clustering_models = NUM_alg * [main()]#
  for i in range(n):
    labels_parea = cluster_ensembles(labels, 7,'nmf', 150, False) #'hgpa', 'mcla', and 'nmf'
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

silh_ce, result_ce = best_clustering(20)

np.array(result_ce)

silh_ce

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

