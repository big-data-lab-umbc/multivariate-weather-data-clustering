import pandas as pd
import numpy as np
import xarray as xr
import netCDF4 as nc
from sklearn.metrics import silhouette_samples

################################### RMSE ###################################

### Non-normalized
# Function that creates a dictionary that holds the values of dates in each cluster
def get_datewise_clusters(formed_clusters): # classification
  Dates_Cluster = {}
  for i in set(formed_clusters): # classification
    Dates_Cluster['Dates_Cluster'+str(i)] = trans_data.index[trans_data.Cluster == i].to_list()
  return Dates_Cluster


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


# Function that creates a dictionary that holds all the cluster centers
def n_nor_get_cluster_centers(input,formed_clusters): #classification
  Cluster_Centers = {}
  Clusters = n_nor_get_clusters(input,formed_clusters)
  for i in set(formed_clusters):
    Cluster_Centers['Cluster_Center' + str(i)] = np.mean(Clusters['Cluster' + str(i)],axis=0)
  return Cluster_Centers


def handle_missing_values(input):
  var_mean = {}
  for i in input.data_vars:
    if input[i].isnull().sum().item() > 0:
      print(i,'has null values')
      var_mean[str(i) + '_mean'] = input[i].mean().item()
      input[i] = input[i].fillna(var_mean[str(i) + '_mean'])
  return input


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



### Normalized

# Function that creates two dictionaries that hold all the clusters and cluster centers
def nor_get_clusters_and_centers(input,formed_clusters):
  Clusters = {}
  Cluster_Centers = {}
  for i in set(formed_clusters):
    Clusters['Cluster' + str(i)] = np.array(input[input.Cluster == i].drop(columns=['Cluster']))
    Cluster_Centers['Cluster_Center' + str(i)] = np.mean(Clusters['Cluster' + str(i)],axis=0)
  return Clusters,Cluster_Centers

# Intra RMSE Calculation Function
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

# Function that creates a dictionary that holds all the cluster centers
def n_nor_get_cluster_centers(input,formed_clusters): #classification
  Cluster_Centers = {}
  Clusters = n_nor_get_clusters(input,formed_clusters)
  for i in set(formed_clusters):
    Cluster_Centers['Cluster_Center' + str(i)] = np.mean(Clusters['Cluster' + str(i)],axis=0)
  return Cluster_Centers

### RMSE Calculation ###

def RMSE(input,formed_clusters,frame,normalize=False):
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

  
    Clusters, Cluster_Centers = nor_get_clusters_and_centers(frame,formed_clusters)

    # Doing the below step after finding the cluster centers. Otherwise, we'll be calculating mean on date (index) too.
    #trans_data = trans_data.reset_index()

  
    mat = pd.DataFrame(columns=range(len(Clusters)),index=range(len(Clusters)))
    for i in range(len(Clusters)):
      avg_cluster['avg_cluster'+str(i)] = np.mean(Clusters['Cluster' + str(i)],axis=0)
    for i in range(len(Clusters)):
      for j in range(len(Clusters)):
        if i == j:
          a = nor_intra_rmse(frame,formed_clusters)
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

################################### End of RMSE ###################################

########################### Spatial Correlation Coefficient ###########################

#For Spatial Correlation coefficient computation
'''
   Input parameters: 
   X : An array, A 2-D array containing multiple variables and observations. 
   Y : An array, A 2-D array containing multiple variables and observations. 
   Each row of x represents a variable, and each column a single observation of all those variables.

   Returns: Pearson product-moment correlation coefficients.:A scaler quantity [-1, 1] 
   where -1 means the inputs are inversly correlated(opposite movements) 
   and 1 means directly correlated (unidirectional parallel movement) 
   while 0 means no correlation.
   
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

def corr_space_2d(var1_2d,var2_2d):
    var1= np.reshape(var1_2d,var1_2d.shape[0]*var1_2d.shape[1])
    var2= np.reshape(var2_2d,var2_2d.shape[0]*var2_2d.shape[1])
    R0=np.corrcoef(var1,var2)
    if(R0.shape !=(2,2)):
        #print("corr_space_2d error: corrcoef maxtrix R0 :",R0)
        stop
    R = R0[0,1]
    return R

def Average(lst):
  for i in lst:
    return sum(lst) / len(lst)


#Intra-spatial correlation coefficient Calculation Function
def n_nor_intra_sp_corr(input,formed_clusters):
  mylist = []
  intra_sp_corr = []
  Clusters = n_nor_get_clusters(input,formed_clusters)

  for i in range(len(Clusters)):
    for j in range(len(Clusters['Cluster' + str(i)])):
      for k in range(len(Clusters['Cluster' + str(i)])):
        corr_coeff = pearson_PM(Clusters['Cluster' + str(i)][j],  Clusters['Cluster' + str(i)][k])
        mylist.append(corr_coeff)

    corr = Average(mylist)
    intra_sp_corr.append(corr)
  return intra_sp_corr

#Inter-spatial correlation coefficient Calculation Function
def n_nor_inter_sp_corr1(input,formed_clusters):
  mylist = []
  inter_sp_corr = []
  Clusters = n_nor_get_clusters(input,formed_clusters)

  for i in range(len(Clusters)):
    for j in range(len(Clusters)):
      for k in range(len(Clusters['Cluster' + str(i)])):
        for l in range(len(Clusters['Cluster' + str(j)])):
          if i != j:
            corr_coeff = pearson_PM(Clusters['Cluster' + str(i)][k],  Clusters['Cluster' + str(j)][l])
            mylist.append(corr_coeff)

    corr = Average(mylist)
    inter_sp_corr.append(corr)
  return inter_sp_corr


### Normalized Approach

def nor_get_clusters(input,formed_clusters):
  Clusters = {}
  for i in set(formed_clusters):
    Clusters['Cluster' + str(i)] = np.array(input[input.Cluster == i].drop(columns=['Cluster']))
  return Clusters

#Intra-spatial correlation coefficient Calculation Function
def nor_intra_sp_corr(input,formed_clusters):
  ''' Rather than passing transformed data to the function, you could use the below steps to transform the data first and later add 
      'Cluster' as a column to the transformed data'''
  # input = datatransformation(input)
  # input = datanormalization(input)
  # input['Cluster'] = formed_clusters
  mylist1 = []
  intra_sp_corr = []
  Clusters = nor_get_clusters(input,formed_clusters)

  for i in range(len(Clusters)):
    for j in range(len(Clusters['Cluster' + str(i)])):
      for k in range(len(Clusters['Cluster' + str(i)])):
        corr_coeff = pearson_PM(Clusters['Cluster' + str(i)][j],  Clusters['Cluster' + str(i)][k])
        mylist1.append(corr_coeff)
        
    corr = Average(mylist1)
    intra_sp_corr.append(corr)
  return intra_sp_corr

#Inter-spatial correlation coefficient Calculation Function
def nor_inter_sp_corr1(input,formed_clusters):
  # input = datatransformation(input)
  # input = datanormalization(input)
  # input['Cluster'] = formed_clusters
  mylist = []
  inter_sp_corr = []
  Clusters = nor_get_clusters(input,formed_clusters)

  for i in range(len(Clusters)):
    for j in range(len(Clusters)):
      for k in range(len(Clusters['Cluster' + str(i)])):
        for l in range(len(Clusters['Cluster' + str(j)])):
          if i != j:
            corr_coeff = pearson_PM(Clusters['Cluster' + str(i)][k],  Clusters['Cluster' + str(j)][l])
            mylist.append(corr_coeff)
            
    corr = Average(mylist)
    inter_sp_corr.append(corr)
  return inter_sp_corr

##### Spat_Corr Calculation 
def Spat_Corr(input,formed_clusters,trans_data,normalize=False):
  inter_sp_corr = []
  mylist = []

  if normalize == False:
    input = handle_missing_values(input)
    Clusters = n_nor_get_clusters(input,formed_clusters)
    mat = pd.DataFrame(columns=range(len(Clusters)),index=range(len(Clusters)))

    for i in range(len(Clusters)):
      for j in range(len(Clusters)):
        for k in range(len(Clusters['Cluster' + str(i)])):
          for l in range(len(Clusters['Cluster' + str(j)])):
            if i == j:
              a = n_nor_intra_sp_corr(input,formed_clusters)
              mat[i].iloc[j] = a[i]
            else:
              results = n_nor_inter_sp_corr1(input,formed_clusters)
              mat[i].iloc[j] = Average(results)
        

  else:
    #trans_data = datatransformation(input)

    # Data Normalization
    #trans_data = datanormalization(trans_data)

    # Adding class centers and cluster numbers as columns to the dataframe
   # trans_data['Cluster'] = classification

    # Rearranging the columns in the dataframe
    #trans_data = trans_data[['Cluster'] + [c for c in trans_data if c not in ['Cluster']]]
  
    Clusters, Cluster_Centers = nor_get_clusters_and_centers(trans_data,formed_clusters)

    #Clusters = n_nor_get_clusters(input,formed_clusters)
    #trans_data = trans_data.reset_index()

  
    mat = pd.DataFrame(columns=range(len(Clusters)),index=range(len(Clusters)))

    for i in range(len(Clusters)):
      for j in range(len(Clusters)):
        for k in range(len(Clusters['Cluster' + str(i)])):
          for l in range(len(Clusters['Cluster' + str(j)])):
            if i == j:
              a = nor_intra_sp_corr(trans_data,formed_clusters)
              mat[i].iloc[j] = a[i]
            else:
              results = nor_inter_sp_corr1(trans_data,formed_clusters)
              mat[i].iloc[j] = Average(results) 
  return mat

########################### End of Spatial Correlation Coefficient ###########################

########################### Silhouette Coefficient ###########################

def silhouette_score1(X, labels, *, metric="euclidean", sample_size=None, random_state=None, **kwds):
    X1=transformd(X)   
    if sample_size is not None:
        X1, labels = check_X_y(X1, labels, accept_sparse=["csc", "csr"])
        random_state = check_random_state(random_state)
        indices = random_state.permutation(X1.shape[0])[:sample_size]
        if metric == "precomputed":
            X1, labels = X1[indices].T[indices].T, labels[indices]
        else:
            X1, labels = X1[indices], labels[indices]
    return np.mean(silhouette_samples(X1, labels, metric=metric, **kwds))

########################### End of Silhouette Coefficient ###########################