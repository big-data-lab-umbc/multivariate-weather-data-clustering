# -*- coding: utf-8 -*-
"""sp_evaluation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rDBz59Bng48omY9EamG5sr38NbhQr0ND

#Internal Cluster Evaluation

RMSE, Spatial Correlation, Silhouette Coefficient, Calinski Harabaz, Davies Bouldin
"""

import pandas as pd
import numpy as np
import xarray as xr
import netCDF4 as nc
from sklearn import preprocessing
from statistics import mean
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score

# from MWDC.preprocessing import datatransformation, datanormalization

# input = datatransformation(input)

# input = datanormalization(input)

# input['Cluster'] = formed_clusters

#The real input to our evaluation functions is a 2-D dataframe with a column for clusters that holds the labels for each record.


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

# Function that creates a dictionary that holds the values of dates in each cluster
def get_datewise_clusters(trans_data, formed_clusters): # classification
  Dates_Cluster = {}
  for i in set(formed_clusters): # classification
    Dates_Cluster['Dates_Cluster'+str(i)] = trans_data.index[trans_data.Cluster == i].to_list()
  return Dates_Cluster


# Function that creats a dictionary that holds all the clusters
def n_nor_get_clusters(input,formed_clusters): # classification
  com_arr = []
  Clusters = {}
  Dates_Cluster = get_datewise_clusters(input, formed_clusters)
  for i in set(formed_clusters):
    for j in Dates_Cluster['Dates_Cluster'+str(i)]:
      arr = np.array(input.isel(time=j).to_array()) # input is data
      com_arr.append(arr)
    Clusters['Cluster' + str(i)] = np.array(com_arr)
    com_arr = []
  return Clusters

# Function that creates a dictionary that holds all the cluster centers
def n_nor_get_cluster_centers(input,formed_clusters): #classification
  Cluster_Centers = {}
  Clusters = n_nor_get_clusters(input,formed_clusters)
  for i in set(formed_clusters):
    Cluster_Centers['Cluster_Center' + str(i)] = np.mean(Clusters['Cluster' + str(i)],axis=0)
  return Cluster_Centers


# Non-normalized
def handle_missing_values(input):
  var_mean = {}
  for i in input.data_vars:
    if input[i].isnull().sum().item() > 0:
      print(i,'has null values')
      var_mean[str(i) + '_mean'] = input[i].mean().item()
      input[i] = input[i].fillna(var_mean[str(i) + '_mean'])
  return input


###########################  RMSE Computation  #######################################



def st_rmse(input, formed_clusters):

  '''
  input: 
        datatype: 4-D spatio-temporal xarray

        formed_clusters: 1-D array of cluster labels classifying each data point along the time dimension
                         to a cluster label

  Output:
         
        An N X M matrix whose diagonal is a measure of intra-rmse between data points in a cluster
        while the rest of the values represent the inter-rmse between data points in different clusters.
     
  '''
  

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

    # if normalize == False:
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
          #print('Inter RMSE between cluster',i,'and cluster',j,'is:')
          

    else:
      trans_data = datatransformation(input)

      # Data Normalization
      trans_data = datanormalization(trans_data)

      # Adding class centers and cluster numbers as columns to the dataframe
      trans_data['Cluster'] = formed_clusters

      # Rearranging the columns in the dataframe
      trans_data = trans_data[['Cluster'] + [c for c in trans_data if c not in ['Cluster']]]
    
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
  return RMSE(input, formed_clusters)


###########################  Spatial Correlation Computation  #######################################



def st_spat_corr(input, formed_clusters):

  '''
  input: 
        datatype: 4-D spatio-temporal xarray

        formed_clusters: 1-D array of cluster labels classifying each data point along the time dimension
                         to a cluster label

  Output:
         
        An N X M matrix whose diagonal is a measure of intra-spatial correlation between data points in a cluster
        while the rest of the values represent the inter-spatial correlation between data points in different clusters.
     
  '''

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

    # Non-normalized
  def handle_missing_values(input):
    var_mean = {}
    for i in input.data_vars:
      if input[i].isnull().sum().item() > 0:
        print(i,'has null values')
        var_mean[str(i) + '_mean'] = input[i].mean().item()
        input[i] = input[i].fillna(var_mean[str(i) + '_mean'])
    return input

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
      input = handle_missing_values(input)
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
      trans_data = datatransformation(input)

      # Data Normalization
      trans_data = datanormalization(trans_data)

      # Adding class centers and cluster numbers as columns to the dataframe
      trans_data['Cluster'] = formed_clusters

      # Rearranging the columns in the dataframe
      trans_data = trans_data[['Cluster'] + [c for c in trans_data if c not in ['Cluster']]]
    
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

  return sp_corr(input,formed_clusters)


###########################  Calinski harabaz Computation  #######################################


def st_calinski(input,formed_clusters):

    # Non-normalized 
   #Intra calinski Calculation Function
  def n_nor_intra_calinski(input,formed_clusters):
    sq_diff = []
    intra_calinski = []
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
      intra_calinski.append(Sqrt_diff_sum)
    return intra_calinski


    # Non-normalized 
    #Intra Calculation Function
  def n_nor_intra_CH(input,formed_clusters):
    sq_diff = []
    intra_CH = []
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
      intra_CH.append(Sqrt_diff_sum)
    return intra_CH


  # Normalized

  def nor_intra_CH(input,formed_clusters):
    intra_CH = []
    sq_diff = []
    #Clusters = n_nor_get_clusters(input,formed_clusters)
    #Cluster_Centers = n_nor_get_cluster_centers(input,formed_clusters)
    Clusters,Cluster_Centers = n_nor_get_cluster_centers(input,formed_clusters)
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
    #Cluster_Centers = n_nor_get_cluster_centers(input,formed_clusters)
    Clusters,Cluster_Centers = n_nor_get_cluster_centers(input,formed_clusters)
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
  def calinski_harabasz(input,formed_clusters,normalize=False):
    inter_CH = []
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
            a = n_nor_intra_CH(input,formed_clusters)
            mat[i].iloc[j] = round(a[i],2)
          else:
            b = nor_inter_CH(input,formed_clusters)
            mat[i].iloc[i] = round(b[i],2)
        

    else:
    #   trans_data = datatransformation(input)

    # # Data Normalization
    #   trans_data = datanormalization(trans_data)

    # # Adding class centers and cluster numbers as columns to the dataframe
    # trans_data['Cluster'] = formed_clusters

    # # Rearranging the columns in the dataframe
    #   trans_data = trans_data[['Cluster'] + [c for c in trans_data if c not in ['Cluster']]]
  
      Clusters, Cluster_Centers = nor_get_clusters_and_centers(trans_data,formed_clusters)

    # Doing the below step after finding the cluster centers. Otherwise, we'll be calculating mean on date (index) too.
    #trans_data = trans_data.reset_index()

  
      mat = pd.DataFrame(columns=range(len(Clusters)),index=range(len(Clusters)))
      for i in range(len(Clusters)):
        avg_cluster['avg_cluster'+str(i)] = np.mean(Clusters['Cluster' + str(i)],axis=0)
      for i in range(len(Clusters)):
        for j in range(len(Clusters)):
          if i == j:
            a = n_nor_intra_CH(input,formed_clusters)
            mat[i].iloc[j] = round(a[i],2)
          else:
            b = nor_inter_CH(input,formed_clusters)
            mat[i].iloc[i] = round(b[i],2)
    return mat

  return calinski_harabasz(input,formed_clusters)



###########################  Davies Computation  #######################################



def st_davies(input,formed_clusters, normalize=True):


    # Non-normalized 
  #Intra Davies-Bouldin Calculation Function
  def n_nor_intra_db(input,formed_clusters):
    sq_diff = []
    intra_db = []
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
      intra_db.append(Sqrt_diff_sum)
    return intra_db



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

    if normalize == False:
      input = handle_missing_values(input)
      Clusters = n_nor_get_clusters(input,formed_clusters)
      mat = pd.DataFrame(columns=range(len(Clusters)),index=range(len(Clusters)))
      for i in range(len(Clusters)):
        avg_cluster['avg_cluster'+str(i)] = np.mean(Clusters['Cluster' + str(i)],axis=0)
      for i in range(len(Clusters)):
        for j in range(len(Clusters)):
          if i == j:
            a = n_nor_intra_db(input,formed_clusters)
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
      # trans_data = datatransformation(data)

    # # Data Normalization
    # trans_data = datanormalization(trans_data)

    # # Adding class centers and cluster numbers as columns to the dataframe
    # trans_data['Cluster'] = formed_clusters

    # # Rearranging the columns in the dataframe
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
            a = nor_intra_db(trans_data,formed_clusters)
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

  return davies_b(input,formed_clusters)


###########################  Silhouette Computation  #######################################


def silhouette_score1(input, formed_clusters, *, metric="euclidean", sample_size=None, random_state=None, **kwds):
  # X0=datatransformation(input)
  # X1 = datanormalization(X0)

  X1 = trans_data

  if sample_size is not None:
      X1, formed_clusters = check_X_y(X1, formed_clusters, accept_sparse=["csc", "csr"])
      random_state = check_random_state(random_state)
      indices = random_state.permutation(X1.shape[0])[:sample_size]
      if metric == "precomputed":
          X1, formed_clusters = X1[indices].T[indices].T, formed_clusters[indices]
      else:
          X1, formed_clusters = X1[indices], formed_clusters[indices]
  return np.mean(silhouette_samples(X1, formed_clusters, metric=metric, **kwds))

#return silhouette_score1(input, formed_clusters)

#st_rmse(data, classification)

#st_spat_corr(data, classification)

#st_davies(data, classification)

#silhouette_score1(data, classification)



"""#External Cluster Evaluation

###Require ground truth/domain knowledge or some information of how the clusters need to look like. For example: we need 2 clusters, one representing summer values and the other representing winter values

Gini Index
"""





"""Entropy"""



