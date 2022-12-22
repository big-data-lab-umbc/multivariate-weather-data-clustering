# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
import netCDF4 as nc
from sklearn import preprocessing
from statistics import mean
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import matplotlib.style as style

from mwdc.preprocessing.preprocessing import datatransformation, datanormalization
from mwdc.clustering import *


def st_rmse(input,formed_clusters):

  '''
  input: 
        datatype: 4-D spatio-temporal xarray

        formed_clusters: 1-D array of cluster labels classifying each data point along the time dimension
                         to a cluster label

  Output:
         
        An N X M matrix whose diagonal is a measure of intra-rmse between data points in a cluster
        while the rest of the values represent the inter-rmse between data points in different clusters.
     
  '''

  trans_data = datatransformation(input)

  trans_data = datanormalization(trans_data)

  trans_data['Cluster'] = formed_clusters
    

  # Non-normalized
  # Function that creates a dictionary that holds the values of dates in each cluster
  def get_datewise_clusters(formed_clusters): # classification
    Dates_Cluster = {}
    for i in set(formed_clusters): # classification
      Dates_Cluster['Dates_Cluster'+str(i)] = trans_data.index[trans_data.Cluster == i].to_list()
    return Dates_Cluster

  # Normalized
  # Function that creates two dictionaries that hold all the clusters and cluster centers
  def nor_get_clusters_and_centers(input,formed_clusters):
    Clusters = {}
    Cluster_Centers = {}
    for i in set(formed_clusters):
      Clusters['Cluster' + str(i)] = np.array(input[input.Cluster == i].drop(columns=['Cluster']))
      Cluster_Centers['Cluster_Center' + str(i)] = np.mean(Clusters['Cluster' + str(i)],axis=0)
    return Clusters,Cluster_Centers


  # Normalized

  def nor_intra_rmse(input,formed_clusters):
    intra_rmse = []
    sq_diff = []
    Clusters,Cluster_Centers = nor_get_clusters_and_centers(input,formed_clusters)
    for i in range(len(Clusters)):
      for j in range(len(Clusters['Cluster' + str(i)])):
        diff = Clusters['Cluster' + str(i)][j] - Cluster_Centers['Cluster_Center' + str(i)]
        Sq_diff = (diff**2)
        sq_diff.append(Sq_diff)
      Sq_diff_sum = sum(sum(sq_diff))
      sq_diff = []
      n = len(Clusters['Cluster' + str(i)])
      Sqrt_diff_sum = np.sqrt(Sq_diff_sum/n)
      intra_rmse.append(Sqrt_diff_sum)
    return intra_rmse



  # RMSE Calculation
  def rmse(input,formed_clusters):
    inter_rmse = []
    avg_cluster = {}
    
    Clusters, Cluster_Centers = nor_get_clusters_and_centers(trans_data,formed_clusters)
    
    mat = pd.DataFrame(columns=range(len(Clusters)),index=range(len(Clusters)))
    for i in range(len(Clusters)):
      avg_cluster['avg_cluster'+str(i)] = np.mean(Clusters['Cluster' + str(i)],axis=0)
    for i in range(len(Clusters)):
      for j in range(len(Clusters)):
        if i == j:
          a = nor_intra_rmse(trans_data,formed_clusters)
          mat[i].iloc[j] = round(a[i],2)
        else:
          diff = avg_cluster['avg_cluster' + str(i)] - avg_cluster['avg_cluster' + str(j)]
          Sq_diff = (diff**2)
          #Sq_diff_sum = sum(Sq_diff)
          Sq_diff_sum = sum(Sq_diff)
          #inter_rmse.append(np.sqrt(Sq_diff_sum))
          Sqrt_diff_sum = np.sqrt(Sq_diff_sum)
          mat[i].iloc[j] = round(Sqrt_diff_sum,2)
          #print('Inter RMSE between cluster',i,'and cluster',j,'is:',inter_rmse.pop())

    return mat
  return rmse(input,formed_clusters)



  ###################### Spatial Correlation ########################


def st_corr(input,formed_clusters):

  '''
  input: 
        datatype: 4-D spatio-temporal xarray

        formed_clusters: 1-D array of cluster labels classifying each data point along the time dimension
                         to a cluster label

  Output:
         
        An N X M matrix whose diagonal is a measure of intra-spatial correlation between data points in a cluster
        while the rest of the values represent the inter-spatial correlation between data points in different clusters.
     
  '''

  trans_data = datatransformation(input)

  trans_data = datanormalization(trans_data)

  trans_data['Cluster'] = formed_clusters



  def pearson_PM(x, y):

    #convert format from netcdf to np array
      #x_form = x.to_numpy()
      #y_form = y.to_numpy()

      #Flatten/transform from 2d to 1d
      X_flat = x.flatten()
      Y_flat = y.flatten()

      #Compute correlation matrix
      corr_mat = np.corrcoef(X_flat, Y_flat)

      #Return entry [0,1]
      return corr_mat[0,1]


  # Non-normalized
  # Function that creates a dictionary that holds the values of dates in each cluster
  def get_datewise_clusters(formed_clusters): # classification
    Dates_Cluster = {}
    for i in set(formed_clusters): # classification
      Dates_Cluster['Dates_Cluster'+str(i)] = trans_data.index[trans_data.Cluster == i].to_list()
    return Dates_Cluster

  def nor_get_clusters_and_centers(input,formed_clusters):
    Clusters = {}
    Cluster_Centers = {}
    for i in set(formed_clusters):
      Clusters['Cluster' + str(i)] = np.array(input[input.Cluster == i].drop(columns=['Cluster']))
      Cluster_Centers['Cluster_Center' + str(i)] = np.mean(Clusters['Cluster' + str(i)],axis=0)
    return Clusters,Cluster_Centers


  #Intra-spatial correlation coefficient Calculation Function

  def nor_intra_sp_corr(input,formed_clusters):
    mylist = []
    intra_sp_corr = []
    Clusters,Cluster_Centers = nor_get_clusters_and_centers(input,formed_clusters)
    
    for i in range(len(Clusters)):
      mylist = []
      for j in range(len(Clusters['Cluster' + str(i)])):
        corr_coeff = pearson_PM(Clusters['Cluster' + str(i)][j], Cluster_Centers['Cluster_Center' + str(i)])
        #print('i: {}, j: {}, corr_coeff:{}'.format(i, j, corr_coeff))
        mylist.append(corr_coeff)
        average_corr_coeff = sum(mylist) / len(mylist)
      intra_sp_corr.append(average_corr_coeff)
    return intra_sp_corr


  def sp_corr(input,formed_clusters,normalize=False):
    inter_sp_corr = []
    avg_cluster = {}
    
    Clusters, Cluster_Centers = nor_get_clusters_and_centers(trans_data,formed_clusters)
  
    mat = pd.DataFrame(columns=range(len(Clusters)),index=range(len(Clusters)))
    for i in range(len(Clusters)):
      avg_cluster['avg_cluster'+str(i)] = np.mean(Clusters['Cluster' + str(i)],axis=0)
    for i in range(len(Clusters)):
      for j in range(len(Clusters)):
        if i == j:
          a = nor_intra_sp_corr(trans_data,formed_clusters)
          mat[i].iloc[j] = round(a[i],2)
        else:
          corr_coeff = pearson_PM(avg_cluster['avg_cluster' + str(i)], avg_cluster['avg_cluster' + str(j)])
          mat[i].iloc[j] = corr_coeff
          
    return mat

  return sp_corr(input,formed_clusters)



###################### Calinski-Harabasz Index ########################



def st_calinski(input, formed_clusters):
  '''
  input: 
        datatype: 4-D spatio-temporal xarray

        formed_clusters: 1-D array of cluster labels classifying each data point along the time dimension
                         to a cluster label

  Output:
         
        An N X M matrix whose diagonal is a measure of intra-spatial correlation between data points in a cluster
        while the rest of the values represent the inter-spatial correlation between data points in different clusters.
     
  '''

  trans_data = datatransformation(input)

  trans_data = datanormalization(trans_data)

  trans_data['Cluster'] = formed_clusters



    # Function that creates a dictionary that holds the values of dates in each cluster
  def get_datewise_clusters(formed_clusters): # classification
    Dates_Cluster = {}
    for i in set(formed_clusters): # classification
      Dates_Cluster['Dates_Cluster'+str(i)] = trans_data.index[trans_data.Cluster == i].to_list()
    return Dates_Cluster


    # Normalized
    # Function that creates two dictionaries that hold all the clusters and cluster centers
  def nor_get_clusters_and_centers(input,formed_clusters):
    Clusters = {}
    Cluster_Centers = {}
    for i in set(formed_clusters):
      Clusters['Cluster' + str(i)] = np.array(input[input.Cluster == i].drop(columns=['Cluster']))
      Cluster_Centers['Cluster_Center' + str(i)] = np.mean(Clusters['Cluster' + str(i)],axis=0)
    return Clusters,Cluster_Centers


  def nor_data_centroid(input,formed_clusters): #classification
    data_enters = {}
    Clusters,Cluster_Centers = nor_get_clusters_and_centers(input,formed_clusters)
    for i in range(len(Clusters)):
      data_enters['Cluster_center'+str(i)] = np.mean(Cluster_Centers['Cluster' + str(i)],axis=0)
    return data_enters


  def nor_intra_CH(input,formed_clusters):
    intra_CH = []
    sq_diff = []
    Clusters,Cluster_Centers = nor_get_clusters_and_centers(trans_data,formed_clusters)
    for i in range(len(Clusters)):
      for j in range(len(Clusters['Cluster' + str(i)])):
        diff = Clusters['Cluster' + str(i)][j] - Cluster_Centers['Cluster_Center' + str(i)]
        Sq_diff = (diff**2)
        sq_diff.append(Sq_diff)
      Sq_diff_sum = sum(sum(sq_diff))
      sq_diff = []
      n = len(Clusters['Cluster' + str(i)])
      Sqrt_diff_sum = np.sqrt(Sq_diff_sum/n)
      intra_CH.append(Sqrt_diff_sum)
    return intra_CH


  def nor_inter_CH(input,formed_clusters):
    inter_CH = []
    sq_diffs = []
    Clusters,Cluster_Centers = nor_get_clusters_and_centers(trans_data,formed_clusters)
    data_center = nor_data_centroid(input,formed_clusters)

    for i in range(len(Cluster_Centers)):
      diff = Cluster_Centers['Cluster_Center' + str(i)] - data_center                     
      Sq_diff = (diff**2)
      sq_diffs.append(Sq_diff)
    Sq_diff_sum = sum(sq_diffs)
    sq_diffs = []
    n = len(Cluster_Centers['Cluster_Center' + str(i)] )
    Sum_diff = np.sum(sum(sum(Sq_diff_sum/n)))
    inter_CH.append(Sum_diff)
    return inter_CH


  # Calinski_Harabasz Calculation
  def calinski_harabasz(input,formed_clusters):
    inter_CH = []
    avg_cluster = {}
  
    Clusters, Cluster_Centers = nor_get_clusters_and_centers(trans_data,formed_clusters)

    mat = pd.DataFrame(columns=range(len(Clusters)),index=range(len(Clusters)))
    for i in range(len(Clusters)):
      avg_cluster['avg_cluster'+str(i)] = np.mean(Clusters['Cluster' + str(i)],axis=0)
    for i in range(len(Clusters)):
      for j in range(len(Clusters)):
        if i == j:
          a = nor_intra_CH(input,formed_clusters)
          mat[i].iloc[j] = round(a[i],2)
        else:
          b = nor_inter_CH(input,formed_clusters)
          mat[i].iloc[i] = round(b[i],2)
    return mat
  return calinski_harabasz(input,formed_clusters)



  ###################### Davies-Bouldin Index ########################



def davies_bouldin(input, formed_clusters):
  '''
  input: 
        datatype: 4-D spatio-temporal xarray

        formed_clusters: 1-D array of cluster labels classifying each data point along the time dimension
                         to a cluster label

  Output:
         
        An N X M matrix whose diagonal is a measure of intra-spatial correlation between data points in a cluster
        while the rest of the values represent the inter-spatial correlation between data points in different clusters.
     
  '''
  
   
  trans_data = datatransformation(input)

  trans_data = datanormalization(trans_data)

  trans_data['Cluster'] = formed_clusters


    # Function that creates a dictionary that holds the values of dates in each cluster
  def get_datewise_clusters(formed_clusters): # classification
    Dates_Cluster = {}
    for i in set(formed_clusters): # classification
      Dates_Cluster['Dates_Cluster'+str(i)] = trans_data.index[trans_data.Cluster == i].to_list()
    return Dates_Cluster

    # Normalized
    # Function that creates two dictionaries that hold all the clusters and cluster centers
  def nor_get_clusters_and_centers(input,formed_clusters):
    Clusters = {}
    Cluster_Centers = {}
    for i in set(formed_clusters):
      Clusters['Cluster' + str(i)] = np.array(input[input.Cluster == i].drop(columns=['Cluster']))
      Cluster_Centers['Cluster_Center' + str(i)] = np.mean(Clusters['Cluster' + str(i)],axis=0)
    return Clusters,Cluster_Centers


    # Normalized

  def nor_intra_db(input,formed_clusters):
    intra_db = []
    sq_diff = []
    Clusters,Cluster_Centers = nor_get_clusters_and_centers(input,formed_clusters)
    for i in range(len(Clusters)):
      for j in range(len(Clusters['Cluster' + str(i)])):
        diff = Clusters['Cluster' + str(i)][j] - Cluster_Centers['Cluster_Center' + str(i)]
        Sq_diff = (diff**2)
        sq_diff.append(Sq_diff)
      Sq_diff_sum = sum(sum(sq_diff))
      sq_diff = []
      n = len(Clusters['Cluster' + str(i)])
      Sqrt_diff_sum = np.sqrt(Sq_diff_sum/n)
      intra_db.append(Sqrt_diff_sum)
    return intra_db



    # Davies-Bouldin Calculation
  def davies_b(input,formed_clusters):
    inter_rmse = []
    avg_cluster = {}
  
    Clusters, Cluster_Centers = nor_get_clusters_and_centers(trans_data,formed_clusters)

  
    mat = pd.DataFrame(columns=range(len(Clusters)),index=range(len(Clusters)))
    for i in range(len(Clusters)):
      avg_cluster['avg_cluster'+str(i)] = np.mean(Clusters['Cluster' + str(i)],axis=0)
    for i in range(len(Clusters)):
      for j in range(len(Clusters)):
        if i == j:
          a = nor_intra_db(trans_data,formed_clusters)
          mat[i].iloc[j] = round(a[i],2)
        else:
          diff = avg_cluster['avg_cluster' + str(i)] - avg_cluster['avg_cluster' + str(j)]
          Sq_diff = (diff**2)
          Sq_diff_sum = sum(Sq_diff)
          Sqrt_diff_sum = np.sqrt(Sq_diff_sum)
          mat[i].iloc[j] = round(Sqrt_diff_sum,2)

    return mat
  return davies_b(input,formed_clusters)


###################### Silhouette Computation Index ########################



def silhouette_score_test(X,Range,menu):
  # X is the dataset
  # Range is the range of clusters
  # menu is the select the type of algorithm needed to run.

  range_n_clusters  = [*range(2,Range, 1)]
  #range_n_clusters = [2, 3, 4, 5, 6]
  silhouette_avg_n_clusters = []
  for n_clusters in range_n_clusters:
      # Create a subplot with 1 row and 2 columns
      fig, (ax1) = plt.subplots(1)
      fig.set_size_inches(18, 7)

      # The 1st subplot is the silhouette plot
      # The silhouette coefficient can range from -1, 1 but in this example all
      # lie within [-0.1, 1]
      ax1.set_xlim([-0.1, 1])
    
      # The (n_clusters+1)*10 is for inserting blank space between silhouette
      # plots of individual clusters, to demarcate them clearly.
      ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

      if menu == 1:
        clusterer = KMeans(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(X)
      elif menu == 2:
         frame1, cluster_labels = kmedoids(n_clusters,X)
      elif menu == 3:
           frame1, cluster_labels = dbscanreal(X)

      # The silhouette_score gives the average value for all the samples.
      # This gives a perspective into the density and separation of the formed
      # clusters
      silhouette_avg = silhouette_score(X, cluster_labels)
      print("For n_clusters =", n_clusters,"The average silhouette_score is :", silhouette_avg)

      silhouette_avg_n_clusters.append(silhouette_avg)
      # Compute the silhouette scores for each sample
      sample_silhouette_values = silhouette_samples(X, cluster_labels)

      y_lower = 10
      for i in range(n_clusters):
          # Aggregate the silhouette scores for samples belonging to
          # cluster i, and sort them
          ith_cluster_silhouette_values = \
              sample_silhouette_values[cluster_labels == i]

          ith_cluster_silhouette_values.sort()

          size_cluster_i = ith_cluster_silhouette_values.shape[0]
          y_upper = y_lower + size_cluster_i

          color = cm.nipy_spectral(float(i) / n_clusters)
          ax1.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)

          # Label the silhouette plots with their cluster numbers at the middle
          ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

          # Compute the new y_lower for next plot
          y_lower = y_upper + 10  # 10 for the 0 samples

      ax1.set_title("The silhouette plot for the various clusters.")
      ax1.set_xlabel("The silhouette coefficient values")
      ax1.set_ylabel("Cluster label")

      # The vertical line for average silhouette score of all the values
      ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

      ax1.set_yticks([])  # Clear the yaxis labels / ticks
      ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

      plt.suptitle(("Silhouette analysis for the clustering on  data " "with n_clusters = %d" % n_clusters), fontsize=14, fontweight='bold')

  plt.show()


def compute_silhouette_score(X, labels,transformation=False, *, metric="euclidean", sample_size=None, random_state=None, **kwds):
    
    """Compute the mean Silhouette Coefficient of all samples.
    The Silhouette Coefficient is calculated using the mean intra-cluster
    distance (``a``) and the mean nearest-cluster distance (``b``) for each
    sample.  The Silhouette Coefficient for a sample is ``(b - a) / max(a,
    b)``.  To clarify, ``b`` is the distance between a sample and the nearest
    cluster that the sample is not a part of.
    Note that Silhouette Coefficient is only defined if number of labels
    is ``2 <= n_labels <= n_samples - 1``.
    This function returns the mean Silhouette Coefficient over all samples.
    To obtain the values for each sample, use :func:`silhouette_samples`.
    The best value is 1 and the worst value is -1. Values near 0 indicate
    overlapping clusters. Negative values generally indicate that a sample has
    been assigned to the wrong cluster, as a different cluster is more similar.
    Read more in the :ref:`User Guide <silhouette_coefficient>`.
    Parameters
    ----------
    X : Can be transformed data or it can be .nc file, as can be transformed within this function.
        When data constains .nc file perform transformation=True.
    labels : array-like of shape (n_samples,)
        Predicted labels for each sample.
    transformation: If you want to push the data through data transformation 
                  transformation=True. 
                  Default ='False'
    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by :func:`metrics.pairwise.pairwise_distances
        <sklearn.metrics.pairwise.pairwise_distances>`. If ``X`` is
        the distance array itself, use ``metric="precomputed"``.
    sample_size : int, default=None
        The size of the sample to use when computing the Silhouette Coefficient
        on a random subset of the data.
        If ``sample_size is None``, no sampling is used.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for selecting a subset of samples.
        Used when ``sample_size is not None``.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.
    **kwds : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a scipy.spatial.distance metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.
    Returns
    -------
    silhouette : float
        Mean Silhouette Coefficient for all samples.
    References
    ----------
    .. [1] `Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
       Interpretation and Validation of Cluster Analysis". Computational
       and Applied Mathematics 20: 53-65.
       <https://www.sciencedirect.com/science/article/pii/0377042787901257>`_
    .. [2] `Wikipedia entry on the Silhouette Coefficient
           <https://en.wikipedia.org/wiki/Silhouette_(clustering)>`_
    """
    if transformation==True:
       trans_data = datatransformation(X)
       trans_data = datanormalization(trans_data)
       if sample_size is not None:
          trans_data, labels = check_X_y(trans_data, labels, accept_sparse=["csc", "csr"])
          random_state = check_random_state(random_state)
          indices = random_state.permutation(trans_data.shape[0])[:sample_size]
          if metric == "precomputed":
              trans_data, labels = trans_data[indices].T[indices].T, labels[indices]
          else:
              trans_data, labels = trans_data[indices], labels[indices]
    else:  
      if sample_size is not None:
          X, labels = check_X_y(X, labels, accept_sparse=["csc", "csr"])
          random_state = check_random_state(random_state)
          indices = random_state.permutation(X.shape[0])[:sample_size]
          if metric == "precomputed":
              X, labels = X[indices].T[indices].T, labels[indices]
          else:
              X, labels = X[indices], labels[indices]
    return np.mean(silhouette_samples(X, labels, metric=metric, **kwds))
  
  
  
  
  
############################RMSE Function using Numpy Array Data-Preprocessing#################################################

###### (Only takes 10-15 Sec, result is same as old function)
from mwdc.preprocessing.preprocessing import data_preprocessing

def st_rmse_omar(data_path,formed_clusters):

  '''
  input: 
        1. "data_path" is the path of the netCDF4 dataset file. (data_path = '/content/multivariate-weather-data-clustering/data/ERA5_meteo_sfc_2021_daily.nc')
        2. "formed_clusters": 1-D array of cluster labels classifying each data point.
  Output:
         
        An N X M matrix whose diagonal is a measure of intra-rmse between data points in a cluster
        while the rest of the values represent the inter-rmse between data points in different clusters.
        
  '''

  #trans_data = datatransformation(input)

  #trans_data = datanormalization(trans_data)
  processed_data = data_preprocessing(data_path, [ ])
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


  # Normalized

  def nor_intra_rmse(input,formed_clusters):
    intra_rmse = []
    sq_diff = []
    Clusters,Cluster_Centers = nor_get_clusters_and_centers(input,formed_clusters)
    for i in range(len(Clusters)):
      for j in range(len(Clusters['Cluster' + str(i)])):
        diff = Clusters['Cluster' + str(i)][j] - Cluster_Centers['Cluster_Center' + str(i)]
        Sq_diff = (diff**2)
        sq_diff.append(Sq_diff)
      Sq_diff_sum = sum(sum(sq_diff))
      sq_diff = []
      n = len(Clusters['Cluster' + str(i)])
      Sqrt_diff_sum = np.sqrt(Sq_diff_sum/n)
      intra_rmse.append(Sqrt_diff_sum)
    return intra_rmse



  # RMSE Calculation
  def rmse(input,formed_clusters):
    inter_rmse = []
    avg_cluster = {}
    
    Clusters, Cluster_Centers = nor_get_clusters_and_centers(trans_data,formed_clusters)
    
    mat = pd.DataFrame(columns=range(len(Clusters)),index=range(len(Clusters)))
    #for i in range(len(Clusters)):
    #  avg_cluster['avg_cluster'+str(i)] = np.mean(Clusters['Cluster' + str(i)],axis=0)
    for i in range(len(Clusters)):
      for j in range(len(Clusters)):
        if i == j:
          a = nor_intra_rmse(trans_data,formed_clusters)
          mat[i].iloc[j] = round(a[i],2)
        else:
          #diff = avg_cluster['avg_cluster' + str(i)] - avg_cluster['avg_cluster' + str(j)]
          diff = Cluster_Centers['Cluster_Center' + str(i)] - Cluster_Centers['Cluster_Center' + str(j)]
          Sq_diff = (diff**2)
          #Sq_diff_sum = sum(Sq_diff)
          Sq_diff_sum = sum(Sq_diff)
          #inter_rmse.append(np.sqrt(Sq_diff_sum))
          Sqrt_diff_sum = np.sqrt(Sq_diff_sum)
          mat[i].iloc[j] = round(Sqrt_diff_sum,2)
          #print('Inter RMSE between cluster',i,'and cluster',j,'is:',inter_rmse.pop())

    return mat
  return rmse(input,formed_clusters)
