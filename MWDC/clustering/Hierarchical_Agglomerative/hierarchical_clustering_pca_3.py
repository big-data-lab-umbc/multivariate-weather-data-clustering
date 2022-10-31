# -*- coding: utf-8 -*-
"""Hierarchical_Clustering_PCA_3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VeatUFvbtLbTcAwwtkJTY5Slf994jpNS
"""

from google.colab import drive
drive.mount('/content/drive')

"""###Install and Import needed libraries###"""

!pip install "dask[dataframe]"

!pip install cartopy
import cartopy

!pip install proj

!pip install basemap



import os
import sys
import pandas as pd
import numpy as np
import xarray as xr
import netCDF4 as nc
import matplotlib.pyplot as plt
#sys.path.insert(0, '/global/homes/x/xzheng/python_lib/')
from netCDF4 import Dataset
import matplotlib as mpl 
import matplotlib.colors as colors
#os.environ['PROJ_LIB'] = r'/global/cfs/cdirs/e3sm/xzheng/conda/pkgs/proj-7.2.0-h277dcde_2/share/'
from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from sklearn import preprocessing
from statistics import mean
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_samples, silhouette_score

"""###Read our data as an xarray##"""

#path2 = ('/content/drive/MyDrive/Data/mock.nc')
#path2 = ('/content/drive/MyDrive/Private/Image_Similarity/mock1.nc')
#path2 = ('/content/drive/MyDrive/Data/mock_v2.nc')
#path2 = ('/content/drive/MyDrive/Data/mock_v2.1.nc')
#path2 = ('/content/drive/MyDrive/Data/mock_v3.nc')
#path2 = ('/content/drive/MyDrive/Data/mock_v3.1.nc')
#path2 = ('/content/drive/MyDrive/Data/mock_v4.nc')
path2 = ('/content/drive/MyDrive/Data/ERA5_meteo_sfc_2021_daily.nc')
#path2 = ('/content/drive/MyDrive/Data/ERA5_meteo_sfc_2021_daily_smalldomain.nc')
#path2 = ('/content/drive/MyDrive/Data/ERA5_meteo_sfc_2021_hourly.nc')
#path2 = ('/content/drive/MyDrive/Data/ERA5_meteo_sfc_2021_hourly_smalldomain.nc')
data = xr.open_dataset(path2, decode_times=False) #To view the date as integers of 0, 1, 2,....
#data = xr.open_dataset(path2)# decode_times=False) #To view the date as integers of 0, 1, 2,....
#data5 = xr.open_dataset(path2) # To view time in datetime format
data

"""###Preprocessing###


1.   Apply Physics-based NaN imputation
2.   Transform and Normalize data
3.   PCA reduce data




"""

#Function to input NaN values across variables
def null_fill(input):

  dask_df = input.to_dask_dataframe(dim_order=None, set_index=False)
  pd_df = dask_df.compute()
  pd_df1 = pd_df.iloc[:, 3:]
  df2 = pd_df1[pd_df1.isnull().any(axis=1)]
  lst = list(df2.index.values)
  df2.loc[:] = np.nan
  dt = pd.concat([pd_df1, df2], axis=0)
  dt3 = dt[~dt.index.duplicated(keep='last')]
  dt4 = dt3[['sst', 'sp', 'u10', 'v10', 'sshf', 'slhf', 't2m']]
  pd_df4 = pd_df.iloc[:, 0:5]
  dff = pd_df4[['time', 'longitude', 'latitude', 'sst']]
  df = pd.merge(dff, dt4, left_index=True, right_index=True).drop('sst_y', axis=1)
  df.rename(columns={'sst_x':'sst'}, inplace=True)
  df_rows = pd.DataFrame(df).set_index(["time", "longitude", "latitude"])
  data = xr.Dataset.from_dataframe(df_rows)
  df_rows = pd.DataFrame(df).set_index(["time", "longitude", "latitude"])
  data = xr.Dataset.from_dataframe(df_rows)

  return data

data = null_fill(data)

def datatransformation(data):
        dask_df = data.to_dask_dataframe(dim_order=None, set_index=False)
        pd_df = dask_df.compute()

        for i in pd_df.columns:
          if pd_df[i].isna().sum() > 0:
            pd_df[i].fillna(value=pd_df[i].mean(), inplace=True)
        
        #col = 'time','lat','lon'
        col = 'time','latitude','longitude'
        fin_df = pd_df.loc[:, ~pd_df.columns.isin(col)]

        trans_data = pd.DataFrame()
        for j in fin_df.columns:
          for i in range(0,pd_df.shape[0]):
              c=(j + '(' + str(pd_df.latitude[i])+','+str(pd_df.longitude[i]) + ')')
              trans_data.loc[pd_df.time[i], c] = pd_df[j][i]

        return trans_data

def datanormalization(input):
  x = input.values #returns a numpy array
  min_max_scaler = preprocessing.MinMaxScaler()
  x_scaled = min_max_scaler.fit_transform(x)
  trans_data = pd.DataFrame(x_scaled, columns=input.columns, index=input.index)
        
  return trans_data

#trans_data1 = datatransformation(data)
#trans_data = datanormalization(trans1_data)

#trans_data.to_csv("/content/drive/MyDrive/Private/Image_Similarity/daily_trans.csv")

#norm_data = datanormalization(trans_data)

#norm_data.to_csv("/content/drive/MyDrive/Private/Image_Similarity/daily_trans_norm.csv")

# trans_data = pd.read_csv("/content/drive/MyDrive/Private/Image_Similarity/daily_trans.csv")
trans_data = pd.read_csv("/content/drive/MyDrive/Private/Image_Similarity/daily_trans_norm.csv")
trans_data = trans_data.iloc[:,1:]
trans_data.head(3)

# norm_data = pd.read_csv("/content/drive/MyDrive/Private/Image_Similarity/daily_trans_norm.csv")
# norm_data = norm_data.iloc[:,1:]
# #norm_data

pca = PCA(n_components=3)
principalComponents = pca.fit_transform(trans_data)
trans_data = pd.DataFrame(data = principalComponents
             , columns = ['PC_1', 'PC_2', 'PC_3'])

trans_data

"""###Clustering algorithm implementation###"""

# Elbow Method for Agglomerative Clustering
model = AgglomerativeClustering()
# k is range of number of clusters.
visualizer = KElbowVisualizer(model, k=(2,20), timings= True)
visualizer.fit(trans_data)        # Fit data to visualizer
visualizer.show()        # Finalize and render figure

# Calinski Harabasz Score for Agglomerative Clustering

model = AgglomerativeClustering()
# k is range of number of clusters.
visualizer = KElbowVisualizer(model, k=(2,20),metric='calinski_harabasz', timings= True)
visualizer.fit(trans_data)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure

# Silhouette Score for Agglomerative Clustering

model = AgglomerativeClustering()
# k is range of number of clusters.
visualizer = KElbowVisualizer(model, k=(2,20),metric='silhouette', timings= True)
visualizer.fit(trans_data)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure

# graph size
plt.figure(1, figsize = (16 ,8))

# creating the dendrogram
dendrogram = sch.dendrogram(sch.linkage(trans_data, method  = "ward"))

var = list(data.variables)

# ploting graphabs
plt.title('Dendrogram')
plt.xlabel(var)
plt.ylabel('Euclidean distances')
plt.show()

"""###Implementing Agglomerative Hierarchical clustering"""

df = pd.read_csv("/content/drive/MyDrive/ECRP_Data_Science/Implementation and Results/Jianwu_test/PCA_Result_Combination/PCA_Combined_var1.csv")

df2 = df.time_step
df2

# calling the agglomerative algorithm and choosing n_clusters = 4 based on elbow value
model = AgglomerativeClustering(n_clusters = 7, affinity = 'euclidean', linkage ='average')

# training the model on transformed data
y_model = model.fit(trans_data)
labels = y_model.labels_

# creating pandas dataframe on transformed data
df1 = pd.DataFrame(df2, columns=['time_step'])
df1['clusterid'] = labels
#df1["cluster"] = cluster.labels_
df1['clusterid'].value_counts()
df1.to_csv("/content/drive/MyDrive/Private/Image_Similarity/Agglo_Daily_PCA3_Norm.csv")
df1

df1.groupby('clusterid').count()

# # creating pandas dataframe on transformed data
# norm_data = pd.DataFrame(norm_data, columns=['time_step'])
# norm_data['Cluster'] = labels
# #df1["cluster"] = cluster.labels_
# norm_data['Cluster'].value_counts()

classification = labels
classification

trans_data['Cluster'] = classification
trans_data

# graph size
plt.figure(1, figsize = (24 ,12))

# creating the dendrogram
dendrogram = sch.dendrogram(sch.linkage(trans_data, method  = "ward"))

plt.axhline(y = 90, color='orange', linestyle ="--")

var = list(data.variables)

# ploting graphabs
plt.title('Dendrogram')
plt.xlabel(var)
plt.ylabel('Euclidean distances')
plt.show()

"""#**Evaluation Metrics**

### Non-normalized functions
"""

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

# Non-normalized
def handle_missing_values(input):
  var_mean = {}
  for i in input.data_vars:
    if input[i].isnull().sum().item() > 0:
      print(i,'has null values')
      var_mean[str(i) + '_mean'] = input[i].mean().item()
      input[i] = input[i].fillna(var_mean[str(i) + '_mean'])
  return input

"""### Normalized functions ###"""

# Normalized
# Function that creates two dictionaries that hold all the clusters and cluster centers
def nor_get_clusters_and_centers(input,formed_clusters):
  Clusters = {}
  Cluster_Centers = {}
  for i in set(formed_clusters):
    Clusters['Cluster' + str(i)] = np.array(input[input.Cluster == i].drop(columns=['Cluster']))
    Cluster_Centers['Cluster_Center' + str(i)] = np.mean(Clusters['Cluster' + str(i)],axis=0)
  return Clusters,Cluster_Centers

"""#**RMSE Computation**

###Intra and Inter RMSE Calculation Function###
"""

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

final1 = RMSE(data,classification,True)
final1

#final1.to_csv("/content/drive/MyDrive/Private/Image_Similarity/rmse_daily_final1.csv")

final2 = RMSE(data,classification,False)
final2

#final2.to_csv("/content/drive/MyDrive/Private/Image_Similarity/rmse_daily_final2.csv")

# This is to showcase the scenario where a user does not pass the third parameter
# The default value has been set to False
final3 = RMSE(data,classification)
final3



"""#**Evaluation - Spatial Correlation Coefficient**"""

#For Spatial Correlation coefficient computation
'''
   Input parameters:
         Data: A 4-D xarray
         labels: 1-D array
         Boolean: Offers the user an option to normalize data

   Returns: 
   
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

# # Converting the normalized data array into a pandas dataframe
# trans_data = datatransformation(data)
# nor_data = np.array(datanormalization(trans_data)) # This is just a trial
# trans_data = pd.DataFrame(nor_data, columns=trans_data.columns, index=trans_data.index)
# # Adding class centers and cluster numbers as columns to the dataframe
# #trans_data['Class_Center'] = class_centers
# trans_data['Cluster'] = classification
# # Rearranging the columns in the dataframe
# #trans_data = trans_data[['Class_Center', 'Cluster'] + [c for c in trans_data if c not in ['Class_Center', 'Cluster']]]
# #trans_data1 = trans_data
# #trans_data = trans_data.reset_index()
# trans_data

#trans_data.to_csv("/content/drive/MyDrive/Private/Image_Similarity/trans_data_daily.csv")

#trans_data = pd.read_csv("/content/drive/MyDrive/Private/Image_Similarity/daily_trans_norm.csv")

#trans_data = trans_data.iloc[:,1:]

#trans_data['Cluster'] = classification
#trans_data

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

"""##Normalized - Spatial Correlation##"""

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
          mat[i].iloc[j] = corr_coeff
          #print('Inter RMSE between cluster',i,'and cluster',j,'is:',inter_rmse.pop())
        

  return mat

sp_final1 = sp_corr(data, classification, True)
sp_final1

#sp_final1.to_csv("/content/drive/MyDrive/Private/Image_Similarity/Hierach_daily_final1.csv")

#sp_final2 = sp_corr(data, classification, False)
#sp_final2

#sp_final2.to_csv("/content/drive/MyDrive/Private/Image_Similarity/Hier_daily_final2.csv")

"""#**Evaluation - Silhouette Coefficient**"""

def silhouette_score1(X, labels, pass_trans_data = True, *, metric="euclidean", sample_size=None, random_state=None, **kwds):
   if pass_trans_data == True:
          k = list(locals().values()) # k will be the list which holds the values of the parameters that are being passed to the function
          #path = str(k[0]) # path will hold the first parameter's value
          #fullpath = os.path.join("/content/drive/MyDrive/Private/Image_Similarity/Transformed_data/" + path + ".csv")
          

          path = ("/content/drive/MyDrive/Private/Image_Similarity/daily_trans_norm.csv")
          #X1 = pd.read_csv(fullpath, index_col=[0]) # saved transformed dataframe will be read
          X1 = pd.read_csv(path, index_col=[0])
          X1 = datanormalization(X1) # the data will be normalized

          if sample_size is not None:
            X1, labels = check_X_y(X1, labels, accept_sparse=["csc", "csr"])
            random_state = check_random_state(random_state)
            indices = random_state.permutation(X1.shape[0])[:sample_size]
            if metric == "precomputed":
              X1, labels = X1[indices].T[indices].T, labels[indices]
            else:
              X1, labels = X1[indices], labels[indices]
          return np.mean(silhouette_samples(X1, labels, metric=metric, **kwds))
   
   else:
     X1 = data[X]
     X1 = datatransformation(X)
     X1 = datanormalization(X1)   
     if sample_size is not None:
        X1, labels = check_X_y(X1, labels, accept_sparse=["csc", "csr"])
        random_state = check_random_state(random_state)
        indices = random_state.permutation(X1.shape[0])[:sample_size]
        if metric == "precomputed":
            X1, labels = X1[indices].T[indices].T, labels[indices]
        else:
            X1, labels = X1[indices], labels[indices]
     return np.mean(silhouette_samples(X1, labels, metric=metric, **kwds))

silhouette_Coefficient = silhouette_score1('data', classification)
print("The average silhouette_score is :", silhouette_Coefficient)

"""#**Calinski-Harabasz Calculation Function**#"""

n_nor_get_cluster_centers(data,classification)

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
  #Centers_sum = np.sum(sum(sum(centers_sum/n)))
  Centers_sum = centers_sum/n
  Center.append(Centers_sum)

  return Center

data_center = data_centroid(data, classification)
data_center

"""#**Intra and Inter Calinski-Harabasz Calculation Function**#
[link text](https://python-bloggers.com/2022/03/calinski-harabasz-index-for-k-means-clustering-evaluation-using-python/)
"""

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
    Sum_diff = np.sum(sum(sum(Sq_diff_sum/n)))
    intra_CH.append(Sum_diff)
  return intra_CH

n_nor_intra_CH(data,classification)

"""### Normalized functions ###"""

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

def nor_intra_CH(input,formed_clusters):
  intra_CH = []
  sq_diff = []
  Clusters = n_nor_get_clusters(input,formed_clusters)
  Cluster_Centers = n_nor_get_cluster_centers(input,formed_clusters)
  #Clusters,Cluster_Centers = nor_get_clusters_and_centers(input,formed_clusters)
  for i in range(len(Clusters)):
    for j in range(len(Clusters['Cluster' + str(i)])):
      diff = Clusters['Cluster' + str(i)][j] - Cluster_Centers['Cluster_Center' + str(i)]
      Sq_diff = (diff**2)
      sq_diff.append(Sq_diff)
    Sq_diff_sum = sum(sum(sq_diff))
    sq_diff = []
    n = len(Clusters['Cluster' + str(i)])
    Sum_diff = np.sum(sum(sum(Sq_diff_sum/n)))
    intra_CH.append(Sum_diff)
  return intra_CH

nor_intra_CH(data,classification)

def nor_inter_CH(input,formed_clusters):
  inter_CH = []
  sq_diffs = []
  Cluster_Centers = n_nor_get_cluster_centers(input,formed_clusters)
  #Clusters,Cluster_Centers = nor_get_clusters_and_centers(input,formed_clusters)
  for i in range(len(Cluster_Centers )):
    diff = Cluster_Centers['Cluster_Center' + str(i)] - data_center
                           
    Sq_diff = (diff**2)
    sq_diffs.append(Sq_diff)
    Sq_diff_sum = sum(sq_diffs)
    sq_diffs = []
    n = len(Cluster_Centers['Cluster_Center' + str(i)] )
    Sum_diff = np.sum(sum(sum(Sq_diff_sum/n)))
    inter_CH.append(Sum_diff)
  return inter_CH

nor_inter_CH(data,classification)

# # Calinski_Harabasz Calculation
# def Calinski_Harabasz(input,formed_clusters,normalize=False):
#   inter_CH = []
#   avg_cluster = {}

#   if normalize == False:
#     input = handle_missing_values(input)
#     Clusters = n_nor_get_clusters(input,formed_clusters)
#     mat = pd.DataFrame(columns=range(len(Clusters)),index=range(len(Clusters)))
#     for i in range(len(Clusters)):
#       avg_cluster['avg_cluster'+str(i)] = np.mean(Clusters['Cluster' + str(i)],axis=0)
#     for i in range(len(Clusters)):
#       for j in range(len(Clusters)):
#         if i == j:
#           a = n_nor_intra_CH(input,formed_clusters)
#           mat[i].iloc[j] = round(a[i],2)
#         else:
#             b = nor_inter_CH(input,formed_clusters)
#             mat[i].iloc[j] = round(b[i],2)
        

#   else:
#     trans_data = datatransformation(input)

#     # Data Normalization
#     trans_data = datanormalization(trans_data)

#     # Adding class centers and cluster numbers as columns to the dataframe
#     trans_data['Cluster'] = formed_clusters

#     # Rearranging the columns in the dataframe
#     trans_data = trans_data[['Cluster'] + [c for c in trans_data if c not in ['Cluster']]]
  
#     Clusters, Cluster_Centers = nor_get_clusters_and_centers(trans_data,formed_clusters)

#     # Doing the below step after finding the cluster centers. Otherwise, we'll be calculating mean on date (index) too.
#     #trans_data = trans_data.reset_index()

  
#     mat = pd.DataFrame(columns=range(len(Clusters)),index=range(len(Clusters)))
#     for i in range(len(Clusters)):
#       avg_cluster['avg_cluster'+str(i)] = np.mean(Clusters['Cluster' + str(i)],axis=0)
#     for i in range(len(Clusters)):
#       for j in range(len(Clusters)):
#         if i == j:
#           a = n_nor_intra_CH(input,formed_clusters)
#           mat[i].iloc[j] = round(a[i],2)
#         else:
#             b = nor_inter_CH(input,formed_clusters)
#             mat[i].iloc[j] = round(b[i],2)

#   return mat

def Calinski_Harabasz(input,formed_clusters):
  inter_CH = []
  avg_cluster = {}
  
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
          mat[i].iloc[j] = round(b[i],2)
  return mat

final = Calinski_Harabasz(data,classification)
final



"""#Visualizations

**Map Visualization with original 2-dimensional dataset**
"""

## read the timestep and it's cluster ID from your clustering results
def read_combined_cluster(csvlink,varid):
    with open(csvlink, mode ='r')as file:
       # reading the CSV file
       csvFile = pd.read_csv(file)
       if(len(varid)>0):    
          id = csvFile['clusterid']
          time_step = csvFile['time_step']
          days = np.zeros(len(id))-999 
 
          for i in range(len(id)):   
              days[i] = int(time_step[i][5:])
                            
                            
    return days,id

##called by plot_map to plot the coastline on the map
def plotcoastline(color='k'):
    lon_c = []
    lat_c = []
    # with open('coast.txt') as f:
    with open('/content/drive/MyDrive/ECRP_Data_Science/Implementation and Results/Jianwu_test/visualizations/coast.txt') as f:
        for line in f:
            data = line.split()
            lon_c.append(float(data[0])-360)
            lat_c.append(float(data[1]))
    plt.plot(lon_c,lat_c,color=color)
    return [lon_c,lat_c]

# plot the map of orignal data (mean value and standard deviation) for each cluster on the specified subpanel
def plot_map(var, var_range,lon0,lat0,fig,panel,cmap0,colorbar,title,ifcontourf):
  ax=plt.subplot(panel)
  if(ifcontourf):  
     p1=plt.contourf(lon0,lat0,var,cmap=cmap0,levels=np.arange(var_range[0],var_range[1],(var_range[1]-var_range[0])/31),extend = 'both') 
     p1.ax.tick_params(labelsize=12)
     plotcoastline(color='k',)
     plt.xlim([min(lon0),max(lon0)])  
     plt.ylim([min(lat0),max(lat0)])    
     plt.title(title,loc='left')   
     plt.xlabel('Longitude')
     plt.ylabel('Latitude')
     if(colorbar):
        ticks = np.linspace(var_range[0], var_range[1], 8, endpoint=True)
        cax = ax.inset_axes([1.04, 0, 0.02, 1], transform=ax.transAxes)
        cb2 = fig.colorbar(p1,orientation='vertical',ax=ax,cax=cax,ticks=ticks)
        cb2.ax.tick_params(labelsize=9)
        
  else:
     p1=ax.contour(lon0,lat0,var,cmap=cmap0,levels=np.arange(var_range[0],var_range[1],(var_range[1]-var_range[0])/11),extend = 'both',linewidth=0.6) 
     p1.ax.tick_params(labelsize=12)
     plt.title(title,loc='right')
     if(colorbar):
        ticks = np.linspace(var_range[0], var_range[1], 12, endpoint=True)
        cax = ax.inset_axes([1.23, 0, 0.02, 1], transform=ax.transAxes)
        cb2 = fig.colorbar(p1,orientation='vertical',ax=ax,cax=cax,ticks=ticks)
        cb2.ax.tick_params(labelsize=9)
  ax.set_aspect(0.65)      

        #cbar = fig.colorbar(p1)
  return [p1]

"""**Specify the location and file name of the raw dataset**"""

# input_dir = './'
# fig_dir = './'
# data_file = '/content/drive/MyDrive/Multivariate Data Independent Study/ERA5_meteo_sfc_2021_daily.nc'

input_dir = '/content/drive/MyDrive/Private/Image_Similarity'
fig_dir = '/content/drive/MyDrive/Data/new_ERA5_meteo_sfc_2021_daily.nc'
year_str='2021'
data_file = '/new_ERA5_meteo_sfc_'+year_str+'_daily.nc'

fcase = input_dir+data_file
##Read time, lat and lon for visualization
fin = Dataset(fcase, "r")
time = np.squeeze(fin['time'][:])
lat0 = np.squeeze(fin['latitude'][:])
lon0 = np.squeeze(fin['longitude'][:])

"""**Specify the variable name and it's unit coefficient here**"""

varids=['sst','slhf','u10','v10','sshf','sp']
ccoefs=[1,-1./3600,1,1,-1./3600,1e-2]

"""**Read the cluster ID assiged to each time step from the clustering results. You might need to modify this cell to fit your csv format**"""

import pandas as pd
# cluster_filename='Kmeans'
# #cluster_filename='PCA_Combined_Norm_Clusters_5'
# cluster_link = cluster_filename+'.csv'

cluster_filename="Agglo_Daily_PCA11_Norm"
cluster_link = input_dir+'/cluster/'+cluster_filename+'.csv'

[days,id]=read_combined_cluster(cluster_link,'OK')


[days,id]=read_combined_cluster(cluster_link,'OK') 
n_cluster = max(id)-min(id)+1
width = 0.3
height = 0.5
panels=[(0.06, 0.08,width, height), (0.39, 0.08,width, height),(0.72, 0.08,width, height),
        (0.06, 0.38, width, height), (0.39, 0.38, width, height),(0.72, 0.38, width, height),
        (0.06, 0.68, width, height),(0.39, 0.68, width, height), (0.72, 0.68, width, height),
]
panels=[(0.06, 0.08,width, height), (0.5, 0.08,width, height),(0.05, 0.5,0.9, 0.3),]


n_cluster = max(id)-min(id)+1
print('total clusters: ',n_cluster)

days

"""**Specify the total subpanels for the visualized clusters. 4 rows x 2 columns for now:**"""

panels=np.arange(421,428,dtype=int) #4 rows 2 columns depending on clusters (how much panels you want?)
#You can choose different colormaps
cmap0='coolwarm'

"""**Start plotting each individual variables: **"""

for ivar in range(len(varids)):
  fig=plt.figure(ivar+1,figsize=[19,22])  
  varid = varids[ivar]  
  var_range=[0,1]
  ccoef = ccoefs[ivar] 
  colorbar = True
  var0 = ccoef*np.squeeze(fin[varid][:])
  ## Fix the range of the values for all of the clusters
  var_range[0]= np.nanmin(var0)+(np.nanmax(var0)-np.nanmin(var0))*0.05
  var_range[1]=np.nanmax(var0)-(np.nanmax(var0)-np.nanmin(var0))*0.05
  print('varid:',varid)
  print('var_range:',var_range)
  ipanel = 0
  for icluster in range(min(id),max(id)+1):
      days_icluster = days[np.where(id==icluster)[0]]
      ndays_icluster = len(days_icluster)
      ## calculate the mean value and standard deviation from the time series of the variable at each (lat,lon)  
      time_icluster = np.zeros(ndays_icluster)
      var_icluster = np.zeros([ndays_icluster,len(lat0),len(lon0)])  
      for iday in range(ndays_icluster):
          istep = np.where(time==days_icluster[iday])
          time_icluster[iday] = time[istep]
          var_icluster[iday]=  np.squeeze(var0[istep])       
      var_mean_icluster = np.nanmean(var_icluster,axis=0)
      var_std_icluster = 2*np.nanstd(var_icluster,axis=0)
           
      title1= 'cluster'+str(icluster)+' n='+ str(len(time_icluster))+' '+varid+' '+ f'{np.nanmean(var_mean_icluster):9.3f}' 
      p = plot_map(var_mean_icluster, var_range,lon0,lat0,fig,panels[ipanel],cmap0,colorbar,title1,ifcontourf=True)
      if(np.nanstd(var_std_icluster)>1e-6*np.nanmean(var_std_icluster)):
          p = plot_map(var_std_icluster, [np.nanmin(var_std_icluster),np.nanmax(var_std_icluster)] ,lon0,lat0,fig,panels[ipanel],'BrBG',colorbar,'2std='+f'{np.nanmean(var_std_icluster):9.3f}',ifcontourf=False)
      else:
          print(np.nanstd(var_std_icluster),np.nanmean(var_std_icluster))
          plt.title('2std='+f'{np.nanmean(var_std_icluster):9.3f}',loc='right')
      ipanel += 1      
  #fig.savefig(varid+'_'+cluster_filename+'.jpg')


#plt.show()

"""**Plot the time series of the cluster ID:**"""

fig=plt.figure(ivar+2,figsize=[18,6])
plt.bar(days,id+0.1,width=0.3)
plt.tick_params(labelsize=12)
plt.xlim([min(days),max(days)])
plt.ylabel('Cluster ID',fontsize=14)
plt.xlabel('Time steps',fontsize=14)