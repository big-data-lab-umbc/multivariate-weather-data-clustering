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


#temp
from sklearn.cluster import KMeans
from MWDC.clustering.KMediods.kmediods import *
from MWDC.clustering.dbscan.dbscan import dbscanreal

from MWDC.preprocessing import datatransformation, datanormalization



#Spatio-Temporal RMSE Function

def ST_RMSE(input,formed_clusters,normalize=False):

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
    
  # Function to get center of dataset (compute the mean value of all centroids)

  # Non-normalized
  # Function that creates a dictionary that holds all the cluster centers
  def data_centroid(input,formed_clusters): #classification
    Cluster_Centers = {}
    centers = []
    Center = []
    Clusters = n_nor_get_clusters(input,formed_clusters)
    for i in set(formed_clusters):
      Cluster_Centers['Cluster_Center' + str(i)] = np.mean(Clusters['Cluster' + str(i)],axis=0)
      centers.append(Cluster_Centers['Cluster_Center' + str(i)])

    centers_sum = sum(centers)
    centers = []
    n = len(Cluster_Centers['Cluster_Center' + str(i)])
    Centers_sum = np.sum(sum(sum(centers_sum/n)))
    Center.append(Centers_sum)

    return Center


  # Non-normalized
  # Function that creates a dictionary that holds the values of dates in each cluster
  def get_datewise_clusters(formed_clusters): # classification
    Dates_Cluster = {}
    for i in set(formed_clusters): # classification
      Dates_Cluster['Dates_Cluster'+str(i)] = trans_data.index[trans_data.Cluster == i].to_list()
    return Dates_Cluster



  # Non-normalized
  # Function that creats a dictionary that holds all the clusters
  def n_nor_get_clusters(input,formed_clusters): # classification
    com_arr = []
    Clusters = {}
    Dates_Cluster = get_datewise_clusters(formed_clusters)
    for i in set(formed_clusters):
      for j in Dates_Cluster['Dates_Cluster'+str(i)]:
        arr = np.array(input.isel(time=j).to_array()) # input is data
        com_arr.append(arr)
      Clusters['Cluster' + str(i)] = np.array(com_arr)
      com_arr = []
    return Clusters


  # Non-normalized
  # Function that creates a dictionary that holds all the cluster centers
  def n_nor_get_cluster_centers(input,formed_clusters): #classification
    Cluster_Centers = {}
    Clusters = n_nor_get_clusters(input,formed_clusters)
    for i in set(formed_clusters):
      Cluster_Centers['Cluster_Center' + str(i)] = np.mean(Clusters['Cluster' + str(i)],axis=0)
    return Cluster_Centers



  # Non-normalized 
  #Intra RMSE Calculation Function
  def n_nor_intra_rmse(input,formed_clusters):
    sq_diff = []
    intra_rmse = []
    Clusters = n_nor_get_clusters(input,formed_clusters)
    Cluster_Centers = n_nor_get_cluster_centers(input,formed_clusters)

    for i in range(len(Clusters)):
      for j in range(len(Clusters['Cluster' + str(i)])):
        diff = Clusters['Cluster' + str(i)][j] - Cluster_Centers['Cluster_Center' + str(i)]
        Sq_diff = (diff**2)
        sq_diff.append(Sq_diff)
      Sq_diff_sum = sum(sq_diff)
      sq_diff = []
      n = len(Clusters['Cluster' + str(i)])
      Sqrt_diff_sum = np.sqrt(sum(sum(sum(Sq_diff_sum/n))))
      intra_rmse.append(Sqrt_diff_sum)
    return intra_rmse

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
  def RMSE(input,formed_clusters,normalize=False):
    inter_rmse = []
    avg_cluster = {}

    if normalize == False:
      input = handle_missing_values(input)
      Clusters = n_nor_get_clusters(input,formed_clusters)
      mat = pd.DataFrame(columns=range(len(Clusters)),index=range(len(Clusters)))
      for i in range(len(Clusters)):
        avg_cluster['avg_cluster'+str(i)] = np.mean(Clusters['Cluster' + str(i)],axis=0)
      for i in range(len(Clusters)):
        for j in range(len(Clusters)):
          if i == j:
            a = n_nor_intra_rmse(input,formed_clusters)
            mat[i].iloc[j] = round(a[i],2)
          else:
            diff = avg_cluster['avg_cluster' + str(i)] - avg_cluster['avg_cluster' + str(j)]
            Sq_diff = (diff**2)
            #Sq_diff_sum = sum(Sq_diff)
            Sq_diff_sum = sum(sum(sum(Sq_diff)))
            #inter_rmse.append(np.sqrt(Sq_diff_sum))
            n = len(avg_cluster['avg_cluster'+str(i)][0])
            Sqrt_diff_sum = np.sqrt(Sq_diff_sum/n)
            mat[i].iloc[j] = round(Sqrt_diff_sum,2)
            #print('Inter RMSE between cluster',i,'and cluster',j,'is:',inter_rmse.pop())
          

    else:
      # trans_data = datatransformation(input)

      # # Data Normalization
      # trans_data = datanormalization(trans_data)

      # # Adding class centers and cluster numbers as columns to the dataframe
      # trans_data['Cluster'] = formed_clusters

      # Rearranging the columns in the dataframe
      # trans_data = trans_data[['Cluster'] + [c for c in trans_data if c not in ['Cluster']]]
    
      Clusters, Cluster_Centers = nor_get_clusters_and_centers(trans_data,formed_clusters)

      # Doing the below step after finding the cluster centers. Otherwise, we'll be calculating mean on date (index) too.
      #trans_data = trans_data.reset_index()

    
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
  return RMSE(input,formed_clusters,normalize=True)






#################################################################################################
#################################################################################################







#Spatio-Temporal Correlation


#Spatio-Temporal Correlation


def ST_CORRELATION(input,formed_clusters,normalize=True):

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
  

  # Function to get center of dataset (compute the mean value of all centroids)

  # Non-normalized
  # Function that creates a dictionary that holds all the cluster centers
  def data_centroid(input,formed_clusters): #classification
    Cluster_Centers = {}
    centers = []
    Center = []
    Clusters = n_nor_get_clusters(input,formed_clusters)
    for i in set(formed_clusters):
      Cluster_Centers['Cluster_Center' + str(i)] = np.mean(Clusters['Cluster' + str(i)],axis=0)
      centers.append(Cluster_Centers['Cluster_Center' + str(i)])

    centers_sum = sum(centers)
    centers = []
    n = len(Cluster_Centers['Cluster_Center' + str(i)])
    Centers_sum = np.sum(sum(sum(centers_sum/n)))
    Center.append(Centers_sum)

    return Center


  # Non-normalized
  # Function that creates a dictionary that holds the values of dates in each cluster
  def get_datewise_clusters(formed_clusters): # classification
    Dates_Cluster = {}
    for i in set(formed_clusters): # classification
      Dates_Cluster['Dates_Cluster'+str(i)] = trans_data.index[trans_data.Cluster == i].to_list()
    return Dates_Cluster



  # Non-normalized
  # Function that creats a dictionary that holds all the clusters
  def n_nor_get_clusters(input,formed_clusters): # classification
    com_arr = []
    Clusters = {}
    Dates_Cluster = get_datewise_clusters(formed_clusters)
    for i in set(formed_clusters):
      for j in Dates_Cluster['Dates_Cluster'+str(i)]:
        arr = np.array(input.isel(time=j).to_array()) # input is data
        com_arr.append(arr)
      Clusters['Cluster' + str(i)] = np.array(com_arr)
      com_arr = []
    return Clusters


  # Non-normalized
  # Function that creates a dictionary that holds all the cluster centers
  def n_nor_get_cluster_centers(input,formed_clusters): #classification
    Cluster_Centers = {}
    Clusters = n_nor_get_clusters(input,formed_clusters)
    for i in set(formed_clusters):
      Cluster_Centers['Cluster_Center' + str(i)] = np.mean(Clusters['Cluster' + str(i)],axis=0)
    return Cluster_Centers



  # Non-normalized 
  #Intra-spatial correlation coefficient Calculation Function
  import functools 

  def n_nor_intra_sp_corr(input,formed_clusters):
    mylist = []
    intra_sp_corr = []
    Clusters = n_nor_get_clusters(input,formed_clusters)
    Cluster_Centers = n_nor_get_cluster_centers(input,formed_clusters)
    
    for i in range(len(Clusters)):
      mylist = []
      for j in range(len(Clusters['Cluster' + str(i)])):
        corr_coeff = pearson_PM(Clusters['Cluster' + str(i)][j], Cluster_Centers['Cluster_Center' + str(i)])
        #print('i: {}, j: {}, corr_coeff:{}'.format(i, j, corr_coeff))
        mylist.append(corr_coeff)
        average_corr_coeff = sum(mylist) / len(mylist)
      intra_sp_corr.append(average_corr_coeff)
    return intra_sp_corr

  

    #Intra-spatial correlation coefficient Calculation Function
  import functools 

  def sp_corr(input,formed_clusters):
    avg_cluster = {}
    Clusters = n_nor_get_clusters(input,formed_clusters)
    mat = pd.DataFrame(columns=range(len(Clusters)),index=range(len(Clusters)))
    for i in range(len(Clusters)):
      avg_cluster['avg_cluster'+str(i)] = np.mean(Clusters['Cluster' + str(i)],axis=0)
    for i in range(len(Clusters)):
      for j in range(len(Clusters)):
        if i == j:
          a = n_nor_intra_sp_corr(input,formed_clusters)
          mat[i].iloc[j] = round(a[i],2)
        else:
          corr_coeff = pearson_PM(avg_cluster['avg_cluster' + str(i)], avg_cluster['avg_cluster' + str(j)])
          mat[i].iloc[j] = corr_coeff
            #print('Inter RMSE between cluster',i,'and cluster',j,'is:',inter_rmse.pop())
      return mat

  # Normalized
  # Function that creates two dictionaries that hold all the clusters and cluster centers
  def nor_get_clusters_and_centers(input,formed_clusters):
    Clusters = {}
    Cluster_Centers = {}
    for i in set(formed_clusters):
      Clusters['Cluster' + str(i)] = np.array(input[input.Cluster == i].drop(columns=['Cluster']))
      Cluster_Centers['Cluster_Center' + str(i)] = np.mean(Clusters['Cluster' + str(i)],axis=0)
    return Clusters,Cluster_Centers



  #Normalized - Intra-spatial correlation coefficient Calculation Function
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



      # Non-normalized Spatial Correlation Calculation
  def sp_corr(input,formed_clusters,normalize=False):
    inter_sp_corr = []
    avg_cluster = {}

    if normalize == False:
      #input = handle_missing_values(input)
      Clusters = n_nor_get_clusters(input,formed_clusters)
      mat = pd.DataFrame(columns=range(len(Clusters)),index=range(len(Clusters)))
      for i in range(len(Clusters)):
        avg_cluster['avg_cluster'+str(i)] = np.mean(Clusters['Cluster' + str(i)],axis=0)
      for i in range(len(Clusters)):
        for j in range(len(Clusters)):
          if i == j:
            a = n_nor_intra_sp_corr(input,formed_clusters)
            mat[i].iloc[j] = round(a[i],2)
          else:
            corr_coeff = pearson_PM(avg_cluster['avg_cluster' + str(i)], avg_cluster['avg_cluster' + str(j)])
            mat[i].iloc[j] = corr_coeff
            #print('Inter RMSE between cluster',i,'and cluster',j,'is:',inter_rmse.pop())
          

    else:
      # trans_data = datatransformation(input)

      # # Data Normalization
      # trans_data = datanormalization(trans_data)

      # # Adding class centers and cluster numbers as columns to the dataframe
      # trans_data['Cluster'] = formed_clusters

      # Rearranging the columns in the dataframe
      # trans_data = trans_data[['Cluster'] + [c for c in trans_data if c not in ['Cluster']]]
    
      Clusters, Cluster_Centers = nor_get_clusters_and_centers(trans_data,formed_clusters)

      #Doing the below step after finding the cluster centers. Otherwise, we'll be calculating mean on date (index) too.
      #trans_data = trans_data.reset_index()

    
      mat = pd.DataFrame(columns=range(len(Clusters)),index=range(len(Clusters)))
      for i in range(len(Clusters)):
        avg_cluster['avg_cluster'+str(i)] = np.mean(Clusters['Cluster' + str(i)],axis=0)
      for i in range(len(Clusters)):
        for j in range(len(Clusters)):
          if i == j:
            a = n_nor_intra_sp_corr(input,formed_clusters)
            mat[i].iloc[j] = round(a[i],2)
          else:
            corr_coeff = pearson_PM(avg_cluster['avg_cluster' + str(i)], avg_cluster['avg_cluster' + str(j)])
            mat[i].iloc[j] = round(corr_coeff, 2)
            #print('Inter RMSE between cluster',i,'and cluster',j,'is:',inter_rmse.pop())
          

    return mat

  return sp_corr(input,formed_clusters,True)

#################################################################################################
#################################################################################################
###########################  Silhouette Computation  #######################################

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



# To call the functions, run the following:

# ST_CORRELATION(data,formed_clusters,True)
# ST_RMSE(data,formed_clusters,True)
# compute_silhouette_score(X, labels,transformation=False, *, metric="euclidean", sample_size=None, random_state=None, **kwds)

