
import pandas as pd
import numpy as np
import xarray as xr
import netCDF4 as nc
from sklearn.preprocessing import StandardScaler
import dask.dataframe

### Transforming Data & Standardizing features by removing the mean and scaling to unit variance. ###

#### Transformation function for Daily Data.

def transformddaily(x):
  import dask.dataframe
# Transforming Data
  dask_df = x.to_dask_dataframe(dim_order=None, set_index=False)
  dd = dask_df.compute()
  sst_data_trans = pd.DataFrame()
  t2m_data_trans = pd.DataFrame()
  v10_data_trans = pd.DataFrame()
  u10_data_trans = pd.DataFrame()
  sp_data_trans = pd.DataFrame()
  sshf_data_trans = pd.DataFrame()
  slhf_data_trans = pd.DataFrame()

  for i in range(0,dd.shape[0]):
    b=('sst'+'('+str(dd.latitude[i])+','+str(dd.longitude[i])+')')
    c=('t2m'+'('+str(dd.latitude[i])+','+str(dd.longitude[i])+')')
    d=('v10'+'('+str(dd.latitude[i])+','+str(dd.longitude[i])+')')
    e=('u10'+'('+str(dd.latitude[i])+','+str(dd.longitude[i])+')')
    f=('sp'+'('+str(dd.latitude[i])+','+str(dd.longitude[i])+')')
    g=('sshf'+'('+str(dd.latitude[i])+','+str(dd.longitude[i])+')')
    h=('slhf'+'('+str(dd.latitude[i])+','+str(dd.longitude[i])+')')

    sst_data_trans.loc[dd.time[i], b] = dd.sst[i]
    t2m_data_trans.loc[dd.time[i], c] = dd.t2m[i]
    v10_data_trans.loc[dd.time[i], d] = dd.v10[i]
    u10_data_trans.loc[dd.time[i], e] = dd.u10[i]
    sp_data_trans.loc[dd.time[i], f] = dd.sp[i]
    sshf_data_trans.loc[dd.time[i], g] = dd.sshf[i]
    slhf_data_trans.loc[dd.time[i], h] = dd.slhf[i]
#Removing Null Values
  sst_data_trans1 = sst_data_trans.values.astype(float)
  sst_data_trans1=sst_data_trans.fillna(9999)
  trans_concat = pd.concat([sst_data_trans1, t2m_data_trans, v10_data_trans, u10_data_trans, sp_data_trans, sshf_data_trans, slhf_data_trans ], axis=1)
  scaler = StandardScaler()
  trans_concat_scaled = scaler.fit_transform(trans_concat)
  return trans_concat_scaled


#### Transformation function for Daily Data ONLY SST.
  def ssttransform(x):
    import dask.dataframe
# Transforming Data
  dask_df = x.to_dask_dataframe(dim_order=None, set_index=False)
  dd = dask_df.compute()
  sst_data_trans = pd.DataFrame()

  for i in range(0,dd.shape[0]):
    b=('sst'+'('+str(dd.latitude[i])+','+str(dd.longitude[i])+')')
    sst_data_trans.loc[dd.time[i], b] = dd.sst[i]
#Removing Null Values
  sst_data_trans1 = sst_data_trans.values.astype(float)
  sst_data_trans1=sst_data_trans.fillna(9999)
  trans_concat = pd.concat([sst_data_trans1 ], axis=1)
  scaler = StandardScaler()
  trans_concat_scaled = scaler.fit_transform(trans_concat)
  return trans_concat_scaled


#### Transformation function for Mock Data.

def transformdmock(x):
  import dask.dataframe
# Transforming Data
  dask_df = x.to_dask_dataframe(dim_order=None, set_index=False)
  dd = dask_df.compute()
  sst_data_trans = pd.DataFrame()
  t2m_data_trans = pd.DataFrame()

  for i in range(0,dd.shape[0]):
    b=('sst'+'('+str(dd.lat[i])+','+str(dd.lon[i])+')')
    c=('t2m'+'('+str(dd.lat[i])+','+str(dd.lon[i])+')')

    sst_data_trans.loc[dd.time[i], b] = dd.sst[i]
    t2m_data_trans.loc[dd.time[i], c] = dd.t2m[i]

#Concating the variables 
  trans_concat = pd.concat([sst_data_trans, t2m_data_trans], axis=1)
  scaler = StandardScaler()
  trans_concat_scaled = scaler.fit_transform(trans_concat)
  return trans_concat_scaled


#### Variable for Quater Map

def transformqm(x):
  import dask.dataframe
# Transforming Data
  dask_df = x.to_dask_dataframe(dim_order=None, set_index=False)
  dd = dask_df.compute()
  t2m_data_trans = pd.DataFrame()
  for i in range(0,dd.shape[0]):
    c=('t2m'+'('+str(dd.latitude[i])+','+str(dd.longitude[i])+')')
    t2m_data_trans.loc[dd.time[i], c] = dd.t2m[i]
#Removing Null Values
  trans_concat = pd.concat([t2m_data_trans ], axis=1)
  scaler = StandardScaler()
  trans_concat_scaled = scaler.fit_transform(trans_concat)
  return trans_concat_scaled


######### Data Transformation ######

def datatransformation(input):
    
    '''This function is used to transform the xarray dataset into a pandas dataframe where the dimension "time" would become the index of the DataFrame and,
      pairs of both dimensions "latitude" and "longitude" will become the columns for each variable'''
    
    # If the given input is a string, the below block will be executed
    if isinstance(input, str) == True:
        data1 = data[input]

        # The below line will convert the xarray into a dask dataframe
        dask_df = data1.to_dask_dataframe(dim_order=None, set_index=False)
        # The below line will convert the dask dataframe into a pandas dataframe
        pd_df = dask_df.compute()

        # The below loop will handle missing values in each and every column of the dataframe by substituting mean of individual columns in the place of the missing values
        for i in pd_df.columns:
          if pd_df[i].isna().sum() > 0:
            pd_df[i].fillna(value=pd_df[i].mean(), inplace=True)
        
        # Since the pandas dataframe (p_df) will also have the dimensions of the xarray as columns, we'll have to remove them. Below code handles that part.
        col = 'time','latitude','longitude'
        fin_df = pd_df.loc[:, ~pd_df.columns.isin(col)]

        # trans_data will be the final dataframe that the function will return.
        trans_data = pd.DataFrame()
        for j in fin_df.columns:
          for i in range(0,pd_df.shape[0]):
              c=(j + '(' + str(pd_df.latitude[i])+','+str(pd_df.longitude[i]) + ')') # Every variable followed by the pairs of latitude and longitude will become the columns
              trans_data.loc[pd_df.time[i], c] = pd_df[j][i] # Based on the column name (var+ (lat,lon)), the correct value of each variable will sit in the right place.

        return trans_data

    # If the input is any thing other than a string, the below code will be executed.
    else:
        # The below line will convert the xarray into a dask dataframe
        dask_df = input.to_dask_dataframe(dim_order=None, set_index=False)
        # The below line will convert the dask dataframe into a pandas dataframe
        pd_df = dask_df.compute()

        # The below loop will handle missing values in each and every column of the dataframe by substituting mean of individual columns in the place of the missing values
        for i in pd_df.columns:
          if pd_df[i].isna().sum() > 0:
            pd_df[i].fillna(value=pd_df[i].mean(), inplace=True)
        
        # Since the pandas dataframe (p_df) will also have the dimensions of the xarray as columns, we'll have to remove them. Below code handles that part.
        col = 'time','latitude','longitude'
        fin_df = pd_df.loc[:, ~pd_df.columns.isin(col)]

        # trans_data will be the final dataframe that the function will return.
        trans_data = pd.DataFrame()
        for j in fin_df.columns:
          for i in range(0,pd_df.shape[0]):
              c=(j + '(' + str(pd_df.latitude[i])+','+str(pd_df.longitude[i]) + ')') # Every variable followed by the pairs of latitude and longitude will become the columns
              trans_data.loc[pd_df.time[i], c] = pd_df[j][i] # Based on the column name (var+ (lat,lon)), the correct value of each variable will sit in the right place.

        return trans_data



######################## physics based #######################

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

  ################## end of physics based ##################

  #################### Data Normilization #################
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def datanormalization(input):
    ''' This function is used to normalize the data that is passed to it. Input in this case will be the transformed pandas dataframe. '''
    x = input.values # returns a numpy array
    min_max_scaler = MinMaxScaler() # calling the function
    x_scaled = min_max_scaler.fit_transform(x) # x_scaled will hold the values of the normalized data
    
    # trans_data will hold the same columns and index of the dataframe that is passed to it. And the values will be the ones saved in x_scaled
    trans_data = pd.DataFrame(x_scaled, columns=input.columns, index=input.index)
        
    return trans_data

################ end of data normalization ################

######## PCA #########

from sklearn.decomposition import PCA

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

######## End of PCA #########