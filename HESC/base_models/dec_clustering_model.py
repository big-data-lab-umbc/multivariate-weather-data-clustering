



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

"""# **Multivariate Climate Data Clustering using Deep Embedding Clustering (DEC) Model**
Here we are dealing with climate data that comprises spatial information, time information, and scientific values. The dataset contains the value of 7 parameters for a region of 41 longitudes and 41 latitudes for 365 days in a year.

Our goal is to create meaningful clusters of 365 days based on the values of these 7 parameters. We have used the model suggested in the paper with the title "Unsupervised deep embedding for clustering analysis".

Source: https://github.com/XifengGuo/DEC-keras/tree/2438070110b17b4fb9bc408c11d776fc1bd1bd56
"""

from time import time
import numpy as np
import keras.backend as K
#from keras.engine.topology import Layer, InputSpec
from tensorflow.keras.layers import Layer, InputSpec
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
from tqdm import trange
#import metrics

import netCDF4 as nc
import pandas as pd
import xarray as xr
import datetime
import datetime as dt
from netCDF4 import date2num,num2date
from math import sqrt

"""##**1. Model Creation**

The Deep Embedding Clustering (DEC) model is created using 4 dense layers. After that, the output of these dense layers is provided to the clustering layer. To train the network the model used the KL divergence loss and Stochastic Gradient Descent (SGD).
"""

def autoencoder(dims, act='relu', init='glorot_uniform'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1
    # input
    x = Input(shape=(dims[0],), name='input')
    h = x

    # internal layers in encoder
    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)

    # hidden layer
    h = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(h)  # hidden layer, features are extracted from here

    y = h
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        y = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)

    # output
    y = Dense(dims[0], kernel_initializer=init, name='decoder_0')(y)

    return Model(inputs=x, outputs=y, name='AE'), Model(inputs=x, outputs=h, name='encoder')


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.
    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DEC(object):
    def __init__(self,
                 dims,
                 n_clusters=10,
                 alpha=1.0,
                 init='glorot_uniform'):

        super(DEC, self).__init__()

        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.autoencoder, self.encoder = autoencoder(self.dims, init=init)

        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(self.encoder.output)
        self.model = Model(inputs=self.encoder.input, outputs=clustering_layer)

    def pretrain(self, x, y=None, optimizer='adam', epochs=200, batch_size=256, save_dir='/content/drive/MyDrive/Jianwu-Wang-Francis-Nji/Papers-by-Francis/Ensemble_Clustering/final/ensemble_alg/DEC_1'):
        print('...Pretraining...')
        self.autoencoder.compile(optimizer=optimizer, loss='mse')

        csv_logger = callbacks.CSVLogger(save_dir + '/pretrain_log.csv')
        cb = [csv_logger]
        if y is not None:
            class PrintACC(callbacks.Callback):
                def __init__(self, x, y):
                    self.x = x
                    self.y = y
                    super(PrintACC, self).__init__()

                def on_epoch_end(self, epoch, logs=None):
                    if int(epochs/10) != 0 and epoch % int(epochs/10) != 0:
                        return
                    feature_model = Model(self.model.input,
                                          self.model.get_layer(
                                              'encoder_%d' % (int(len(self.model.layers) / 2) - 1)).output)
                    features = feature_model.predict(self.x)
                    km = KMeans(n_clusters=len(np.unique(self.y)), n_init=20, n_jobs=4)
                    y_pred = km.fit_predict(features)
                    # print()
                    #print(' '*8 + '|==>  acc: %.4f,  nmi: %.4f  <==|' % (metrics.acc(self.y, y_pred), metrics.nmi(self.y, y_pred)))

            cb.append(PrintACC(x, y))

        # begin pretraining
        t0 = time()
        self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=cb)
        print('Pretraining time: %ds' % round(time() - t0))
        self.autoencoder.save_weights(save_dir + '/ae_weights.h5')
        print('Pretrained weights are saved to %s/ae_weights.h5' % save_dir)
        self.pretrained = True

    def load_weights(self, weights):
        self.model.load_weights(weights)

    def extract_features(self, x):
        return self.encoder.predict(x)

    def predict(self, x):
        q = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, optimizer='sgd', loss='kld'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(self, x, y=None, maxiter=2e4, batch_size=256, tol=1e-3,
            update_interval=140, save_dir='/content/drive/MyDrive/Jianwu-Wang-Francis-Nji/Papers-by-Francis/Ensemble_Clustering/final/ensemble_alg/DEC_1'):

        print('Update interval', update_interval)
        save_interval = 500
        print('Save interval', save_interval)


        t1 = time()
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(self.encoder.predict(x))
        y_pred_last = np.copy(y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        import csv
        logfile = open(save_dir + '/dec_log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter','loss'])
        logwriter.writeheader()

        loss = 0
        index = 0
        index_array = np.arange(x.shape[0])
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                y_pred = q.argmax(1)
                print("#### inside iteration ### ", ite)
                if y is not None:
                    loss = np.round(loss, 5)
                    logdict = dict(iter=ite, loss=loss)
                    logwriter.writerow(logdict)
                    print('Iter %d: ' % (iter), ' ; loss=', loss)

                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                print("##### Prediction in side the iter and the delta_label is ", delta_label)
                y_pred_last = np.copy(y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break

            idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
            loss = self.model.train_on_batch(x=x[idx], y=p[idx])
            print("#### the loss is ", loss)
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

            if ite % save_interval == 0:
                print('saving model to:', save_dir + '/DEC_model_' + str(ite) + '.h5')
                self.model.save_weights(save_dir + '/DEC_model_' + str(ite) + '.h5')

            ite += 1

        logfile.close()
        print('saving model to:', save_dir + '/DEC_model_final.h5')
        self.model.save_weights(save_dir + '/DEC_model_final.h5')

        return y_pred

"""# **2. Data preparation**
The model takes one row for each data point. But our dataset size is 265x7x41x41. So we have transformed the data of one day from 7x41x41 to one row with 11767 columns. The dataset has some NaN values in the SST variable. To replace these NaN values we used the mean value of the full dataset. The function returns one NumPy array with size (365, 11767).
"""

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

def data_preprocessing(data_path):
  rdata_daily = xr.open_dataset(data_path)    # data_path = '/content/drive/MyDrive/ERA5_Dataset.nc'
  rdata_daily_np_array = np.array(rdata_daily.to_array())   # the shape of the dailt data is (7, 365, 41, 41)
  rdata_daily_np_array_T = rdata_daily_np_array.transpose(1,0,2,3)   # transform the dailt data from (7, 365, 41, 41) to (365, 7, 41, 41)
  overall_mean = np.nanmean(rdata_daily_np_array_T[:, :, :, :])
  for i in range(rdata_daily_np_array_T.shape[0]):
    for j in range(rdata_daily_np_array_T.shape[1]):
      for k in range(rdata_daily_np_array_T.shape[2]):
        for l in range(rdata_daily_np_array_T.shape[3]):
          if np.isnan(rdata_daily_np_array_T[i, j, k, l]):
            #print("NAN data in ", i, j, k, l)
            rdata_daily_np_array_T[i, j, k, l] = overall_mean
  rdata_daily_np_array_T_R = rdata_daily_np_array_T.reshape((rdata_daily_np_array_T.shape[0], -1))  # transform the dailt data from (365, 7, 41, 41) to (365, 11767)
  min_max_scaler = preprocessing.MinMaxScaler() # calling the function
  rdata_daily_np_array_T_R_nor = min_max_scaler.fit_transform(rdata_daily_np_array_T_R)   # now normalize the data, otherwise the loss will be very big
  rdata_daily_np_array_T_R_nor = np.float32(rdata_daily_np_array_T_R_nor)    # convert the data type to float32, otherwise the loass will be out-of-limit
  return rdata_daily_np_array_T_R_nor

path = '/content/drive/MyDrive/Jianwu-Wang-Francis-Nji/Papers-by-Francis/Ensemble_Clustering/final/ensemble_alg/Final_ensemble/'

data_path ='/content/drive/MyDrive/Data/ERA5_meteo_sfc_2021_daily.nc'
data_nor = data_preprocessing(data_path)

data_nor.shape

from sklearn.metrics import silhouette_samples, silhouette_score

def silhouette_score1(X, labels, *, metric="cosine", sample_size=None, random_state=None, **kwds):
 return np.mean(silhouette_samples(X, labels, metric=metric, **kwds))

"""## **3. Model Training**
This function defines related parameters to train the model. Then instantiate the model and train on the pre-processed data. The model tries to optimize the clustering loss. After training the model returns the cluster results.
"""

def main():
    # setting the hyper parameters

    batch_size = 128
    maxiter = 2e4
    pretrain_epochs = 3
    update_interval = 30
    tol = 0.000001
    ae_weights = None
    save_dir = '/content/drive/MyDrive/Jianwu-Wang-Francis-Nji/Papers-by-Francis/Ensemble_Clustering/final/ensemble_alg/DEC_1'

    # load dataset
    x = data_nor
    y = None
    n_clusters = 7

    init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')
    pretrain_optimizer = SGD(learning_rate=0.1, momentum=0.9)

    dec = DEC(dims=[x.shape[-1], 500, 500, 2000, 150], n_clusters=n_clusters, init=init)

    if ae_weights is None:
        dec.pretrain(x=x, y=y, optimizer=pretrain_optimizer,
                     epochs=pretrain_epochs, batch_size=batch_size,
                     save_dir=save_dir)
    else:
        dec.autoencoder.load_weights(ae_weights)

    dec.model.summary()
    t0 = time()
    dec.compile(optimizer=SGD(0.000001, 0.9), loss='kld')
    y_pred = dec.fit(x, y=y, tol=tol, maxiter=maxiter, batch_size=batch_size,
                     update_interval=update_interval, save_dir=save_dir)
    print('clustering time: ', (time() - t0))
    return y_pred

n_clusters = 7

"""# **Evaluation Metrics**

**Silhouette Score**
"""

def silhouette_score1(X, labels, *, metric="cosine", sample_size=None, random_state=None, **kwds):
 return np.mean(silhouette_samples(X, labels, metric="cosine", **kwds))

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

results = main()
 results



silh = silhouette_score1(data_nor,  results)
u,indices = np.unique(results,return_counts = True) # sc=0.3412 st 64
# u,indices
print(silh)
print(u,indices)

print("Davies-Bouldin score is ", davies_bouldin_score(data_nor, results))

print("Calinski Harabas score is ", calinski_harabasz_score(data_nor, results))

print("RMSE score is ", total_rmse('/content/drive/MyDrive/Data/ERA5_meteo_sfc_2021_daily.nc', results))

print("Variance is ", avg_var(data_nor, results))

print("Inter-cluster distance ", avg_inter_dist(data_nor, results))

def best_clustering(n):

  n # Number of single models used
    #MIN_PROBABILITY = 0.6 # The minimum threshold of occurances of datapoints in a cluster

  sil = []
  sil_score = []

  for i in range(n):
    results = main()
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

silhouette, result_1 = best_clustering(20)

pickle.dump(result_1, open(path + 'DTC_hom_ens_' + str(silhouette) + '.pkl', "wb"))

silh = silhouette_score1(data_nor,  result_1)
u,indices = np.unique(result_1,return_counts = True) # sc=0.3412 st 64
# u,indices
print(silh)
print(u,indices)

silhouette, result_2 = best_clustering(20)

pickle.dump(result_2, open(path + 'DEC_hom_ens_' + str(silhouette) + '.pkl', "wb"))

silh = silhouette_score1(data_nor,  result_2)
u,indices = np.unique(result_2,return_counts = True) # sc=0.3412 st 64
# u,indices
print(silh)
print(u,indices)

silhouette, result_3 = best_clustering(20)

pickle.dump(result_3, open(path + 'DEC_hom_ens_' + str(silhouette) + '.pkl', "wb"))

data_nor_eval = data_nor

silh = silhouette_score1(data_nor,  result_3)
u,indices = np.unique(result_3,return_counts = True) # sc=0.3412 st 64
# u,indices
print(silh)
print(u,indices)

silhouette, result_4 = best_clustering(20)

pickle.dump(result_4, open(path + 'DEC_hom_ens_' + str(silhouette) + '.pkl', "wb"))

silh = silhouette_score1(data_nor_eval,  result_4)
u,indices = np.unique(result_4,return_counts = True) # sc=0.3412 st 64
# u,indices
print(silh)
print(u,indices)

from sklearn.metrics import davies_bouldin_score

print("Davies-Bouldin score is ", davies_bouldin_score(data_nor, result_4))

from sklearn.metrics import calinski_harabasz_score
ch_index = calinski_harabasz_score(data_nor, result_4)

print(ch_index)

print("RMSE score is ", total_rmse('/content/drive/MyDrive/Data/ERA5_meteo_sfc_2021_daily.nc', result_4))

print("Variance is ", avg_var(data_nor, result_4))

print("Inter-cluster distance ", avg_inter_dist(data_nor, result_4))

silhouette, result_5 = best_clustering(20)

pickle.dump(result_5, open(path + 'DEC_hom_ens_' + str(silhouette) + '.pkl', "wb"))

silh = silhouette_score1(data_nor_eval,  result_5)
u,indices = np.unique(result_5,return_counts = True) # sc=0.3412 st 64
# u,indices
print(silh)
print(u,indices)

from sklearn.metrics import davies_bouldin_score

print("Davies-Bouldin score is ", davies_bouldin_score(data_nor, result_5))

from sklearn.metrics import calinski_harabasz_score
ch_index = calinski_harabasz_score(data_nor, result_5)

print(ch_index)

print("RMSE score is ", total_rmse('/content/drive/MyDrive/Data/ERA5_meteo_sfc_2021_daily.nc', result_5))

print("Variance is ", avg_var(data_nor, result_5))

print("Inter-cluster distance ", avg_inter_dist(data_nor, result_5))

"""# **Homogeneous Ensemble**"""

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

clustering_models = NUM_alg * [ result_1, result_2, result_3, result_4, result_5]


clt_sim_matrix = ClusterSimilarityMatrix() #Initializing the similarity matrix
for model in clustering_models:
  clt_sim_matrix.fit(model)

sim_matrixx = clt_sim_matrix.similarity
sim_matrixx = sim_matrixx/sim_matrixx.diagonal()
sim_matrixx[sim_matrixx < occurrence_threshold] = 0

# norm_sim_matrix[norm_sim_matrix < occurrence_threshold] = 0

unique_labels = np.unique(np.concatenate(clustering_models))

print(sim_matrixx)
np.save('/content/drive/MyDrive/Jianwu-Wang-Francis-Nji/Papers-by-Francis/Ensemble_Clustering/final/ensemble_alg/DEC_ens_co_occurrence_matrix.npy', sim_matrixx)
#print(norm_sim_matrix)

import numpy as geek

data_nor_eval = data_nor

#sim_matrixx = geek.load('/content/drive/MyDrive/Jianwu-Wang-Francis-Nji/Papers-by-Francis/Ensemble_Clustering/final/ensemble_alg/Non-negative Matrix Factorization/DEC_co_occurrence_matrix.npy')

from sklearn.cluster import SpectralClustering
spec_clt = SpectralClustering(n_clusters=7, affinity='precomputed',
                              n_init=5, random_state=214)
final_labels = spec_clt.fit_predict(sim_matrixx)



# result = pickle.load(open("/content/drive/MyDrive/Jianwu-Wang-Francis-Nji/Papers-by-Francis/Ensemble_Clustering/final/ensemble_alg/Final_ensemble/DSC_Ensemble1_032.pkl", "rb"))

"""# **Evaluation Metrics**"""

silh = silhouette_score1(data_nor,  final_labels)
u,indices = np.unique(final_labels,return_counts = True)

print("Silhouette score is ", silh)
print("Cluster index ", u, "and Cluster Sizes: ", indices)

pickle.dump(final_labels, open(path + 'DEC_fin_ens_' + str(silh) + '.pkl', "wb"))

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
ch_index = calinski_harabasz_score(data_nor, result_ce)

print(ch_index)

from sklearn.metrics import calinski_harabasz_score
ch = calinski_harabasz_score(data_nor, result_ce)
print("Davies-Bouldin score is ", ch)

print("RMSE score is ", total_rmse('/content/drive/MyDrive/Data/ERA5_meteo_sfc_2021_daily.nc', result_ce))

print("Variance is ", avg_var(data_nor, result_ce))

print("Inter-cluster distance ", avg_inter_dist(data_nor, result_ce))

