


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

import tensorflow as tf
import numpy as np

!pip install netCDF4

import netCDF4 as nc
import pandas as pd
import numpy as np
import xarray as xr
import datetime
import datetime as dt
from netCDF4 import date2num,num2date
from math import sqrt
from time import time

from google.colab import drive
drive.mount('/content/drive')

## This function will will pre-process our daily data for DEC model as numpy array
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
            rdata_daily_np_array_T[i, j, k, l] = overall_mean  # np.nanmean(rdata_daily_np_array_T[i, j, k, :])
  rdata_daily_np_array_T = rdata_daily_np_array_T.transpose(0,2,3,1)
  rdata_daily_np_array_T_R = rdata_daily_np_array_T.reshape((rdata_daily_np_array_T.shape[0], -1))  # transform the dailt data from (365, 7, 41, 41) to (365, 11767)
  min_max_scaler = preprocessing.MinMaxScaler() # calling the function
  rdata_daily_np_array_T_R_nor = min_max_scaler.fit_transform(rdata_daily_np_array_T_R)   # now normalize the data, otherwise the loss will be very big
  #rdata_daily_np_array_T_R_nor = np.float32(rdata_daily_np_array_T_R_nor)    # convert the data type to float32, otherwise the loass will be out-of-limit
  rdata_daily_np_array_T_R_nor_R = rdata_daily_np_array_T_R_nor.reshape((rdata_daily_np_array_T_R_nor.shape[0], rdata_daily_np_array.shape[2], rdata_daily_np_array.shape[3], rdata_daily_np_array.shape[0]))
  return rdata_daily_np_array_T_R_nor, rdata_daily_np_array_T_R_nor_R

path = '/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/'

data_nor_eval, data_clustering = data_preprocessing('/content/drive/MyDrive/Data/ERA5_meteo_sfc_2021_daily.nc')

data_nor_eval.shape, data_clustering.shape

data_nor = data_nor_eval

import random
from sklearn import metrics
#from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment



def get_batch(batch_size, mnist_data, mnist_labels):
	batch_index = random.sample(range(len(mnist_labels)), batch_size)

	batch_data = np.empty([batch_size, 48, 48, 7], dtype=np.float32)
	batch_label = np.empty([batch_size], dtype=np.int32)
	for n, i in enumerate(batch_index):
		batch_data[n, ...] = mnist_data[i, ...]
		batch_label[n] = mnist_labels[i]

	return batch_data, batch_label


def get_mnist_batch_test(batch_size, mnist_data, i):
	batch_data = np.copy(mnist_data[batch_size*i:batch_size*(i+1), ...])
	# batch_label = np.copy(mnist_labels[batch_size*i:batch_size*(i+1)])

	return batch_data


def get_svhn_batch(batch_size, svhn_data, svhn_labels):
	batch_index = random.sample(range(len(svhn_labels)), batch_size)

	batch_data = np.empty([batch_size, 32, 32, 3], dtype=np.float32)
	batch_label = np.empty([batch_size], dtype=np.int32)
	for n, i in enumerate(batch_index):
		batch_data[n, ...] = svhn_data[i, ...]
		batch_label[n] = svhn_labels[i]

	return batch_data, batch_label


def clustering_acc(y_true, y_pred):
	y_true = y_true.astype(np.int64)
	assert y_pred.size == y_true.size
	D = max(y_pred.max(), y_true.max()) + 1
	w = np.zeros((D, D), dtype=np.int64)
	for i in range(y_pred.size):
		w[y_pred[i], y_true[i]] += 1
	ind = linear_assignment(w.max() - w)

	return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def NMI(y_true,y_pred):
	return metrics.normalized_mutual_info_score(y_true, y_pred)


def ARI(y_true,y_pred):
	return metrics.adjusted_rand_score(y_true, y_pred)

def DACNetwork(in_img, num_cluster, name='dacNetwork', reuse=False):
	with tf.variable_scope(name, reuse=reuse):
		# conv1
		conv1 = tf.layers.conv2d(in_img, 64, [3,3], [1,1], padding='valid', activation=None, kernel_initializer=tf.keras.initializers.he_normal())
		conv1 = tf.layers.batch_normalization(conv1, axis=-1, epsilon=1e-5, training=True, trainable=False)
		conv1 = tf.nn.relu(conv1)
		# conv2
		conv2 = tf.layers.conv2d(conv1, 64, [3,3], [1,1], padding='valid', activation=None, kernel_initializer=tf.keras.initializers.he_normal())
		conv2 = tf.layers.batch_normalization(conv2, axis=-1, epsilon=1e-5, training=True, trainable=False)
		conv2 = tf.nn.relu(conv2)
		# conv3
		conv3 = tf.layers.conv2d(conv2, 64, [3,3], [1,1], padding='valid', activation=None, kernel_initializer=tf.keras.initializers.he_normal())
		conv3 = tf.layers.batch_normalization(conv3, axis=-1, epsilon=1e-5, training=True, trainable=False)
		conv3 = tf.nn.relu(conv3)
		conv3 = tf.layers.max_pooling2d(conv3, [2,2], [2,2])
		conv3 = tf.layers.batch_normalization(conv3, axis=-1, epsilon=1e-5, training=True, trainable=False)
		# conv4
		conv4 = tf.layers.conv2d(conv3, 128, [3,3], [1,1], padding='valid', activation=None, kernel_initializer=tf.keras.initializers.he_normal())
		conv4 = tf.layers.batch_normalization(conv4, axis=-1, epsilon=1e-5, training=True, trainable=False)
		conv4 = tf.nn.relu(conv4)
		# conv5
		conv5 = tf.layers.conv2d(conv4, 128, [3,3], [1,1], padding='valid', activation=None, kernel_initializer=tf.keras.initializers.he_normal())
		conv5 = tf.layers.batch_normalization(conv5, axis=-1, epsilon=1e-5, training=True, trainable=False)
		conv5 = tf.nn.relu(conv5)
		# conv6
		conv6 = tf.layers.conv2d(conv5, 128, [3,3], [1,1], padding='valid', activation=None, kernel_initializer=tf.keras.initializers.he_normal())
		conv6 = tf.layers.batch_normalization(conv6, axis=-1, epsilon=1e-5, training=True, trainable=False)
		conv6 = tf.nn.relu(conv6)
		conv6 = tf.layers.max_pooling2d(conv6, [2,2], [2,2])
		conv6 = tf.layers.batch_normalization(conv6, axis=-1, epsilon=1e-5, training=True, trainable=False)
		# conv7
		conv7 = tf.layers.conv2d(conv6, 10, [1,1], [1,1], padding='valid', activation=None, kernel_initializer=tf.keras.initializers.he_normal())
		conv7 = tf.layers.batch_normalization(conv7, axis=-1, epsilon=1e-5, training=True, trainable=False)
		conv7 = tf.nn.relu(conv7)
		conv7 = tf.layers.average_pooling2d(conv7, [2,2], [2,2])
		conv7 = tf.layers.batch_normalization(conv7, axis=-1, epsilon=1e-5, training=True, trainable=False)
		conv7_flat = tf.layers.flatten(conv7)

		# dense8
		fc8 = tf.layers.dense(conv7_flat, 10, kernel_initializer=tf.initializers.identity())
		fc8 = tf.layers.batch_normalization(fc8, axis=-1, epsilon=1e-5, training=True, trainable=False)
		fc8 = tf.nn.relu(fc8)
		# dense9
		fc9 = tf.layers.dense(fc8, num_cluster, kernel_initializer=tf.initializers.identity())
		fc9 = tf.layers.batch_normalization(fc9, axis=-1, epsilon=1e-5, training=True, trainable=False)
		fc9 = tf.nn.relu(fc9)

		out = tf.nn.softmax(fc9)

	return out

data_clustering.shape

data_nor_eval

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def train():
  tf.reset_default_graph()
  num_cluster = 7
  eps = 1e-10  # term added for numerical stability of log computations
  # ------------------------------------build the computation graph------------------------------------------
  data_pool_input = tf.placeholder(shape=[None, 48, 48, 7], dtype=tf.float32, name='data_pool_input')
  u_thres = tf.placeholder(shape=[], dtype=tf.float32, name='u_thres')
  l_thres = tf.placeholder(shape=[], dtype=tf.float32, name='l_thres')
  lr = tf.placeholder(shape=[], dtype=tf.float32, name='learning_rate')

  # get similarity matrix
  label_feat = DACNetwork(data_pool_input, num_cluster, name='dacNetwork', reuse=False)
  label_feat_norm = tf.nn.l2_normalize(label_feat, dim=1)
  sim_mat = tf.matmul(label_feat_norm, label_feat_norm, transpose_b=True)

  pos_loc = tf.greater(sim_mat, u_thres, name='greater')
  neg_loc = tf.less(sim_mat, l_thres, name='less')
  # select_mask = tf.cast(tf.logical_or(pos_loc, neg_loc, name='mask'), dtype=tf.float32)
  pos_loc_mask = tf.cast(pos_loc, dtype=tf.float32)
  neg_loc_mask = tf.cast(neg_loc, dtype=tf.float32)

  # get clusters
  pred_label = tf.argmax(label_feat, axis=1)

  # define losses and train op
  pos_entropy = tf.multiply(-tf.log(tf.clip_by_value(sim_mat, eps, 1.0)), pos_loc_mask)
  neg_entropy = tf.multiply(-tf.log(tf.clip_by_value(1-sim_mat, eps, 1.0)), neg_loc_mask)

  loss_sum = tf.reduce_mean(pos_entropy) + tf.reduce_mean(neg_entropy)
  train_op = tf.train.RMSPropOptimizer(lr).minimize(loss_sum)

  # -------------------------------------------prepared datasets----------------------------------------------
  train_data = data_clustering
  data_labels = np.zeros(data_clustering.shape[0])

  # --------------------------------------------run the graph-------------------------------------------------
  saver = tf.train.Saver()
  batch_size = 16
  base_lr = 0.01
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    lamda = 0
    epoch = 1
    u = 0.95
    l = 0.455
    while u > l:
      u = 0.95 - lamda
      l = 0.455 + 0.1*lamda
      for i in range(1, int(1001)):  # 1000 iterations is roughly 1 epoch
        data_samples, _ = get_batch(batch_size, train_data, data_labels)
        feed_dict={data_pool_input: data_samples, u_thres: u, l_thres: l, lr: base_lr}
        train_loss, _ = sess.run([loss_sum, train_op], feed_dict=feed_dict)
        if i % 20 == 0:
          print('training loss at iter %d is %f' % (i, train_loss))
      lamda += 1.1 * 0.009

      if epoch % 1000 == 0:  # save model at every 5 epochs
        model_name = 'DAC_ep_' + str(epoch) + str(round(time()))+ '.ckpt'
        save_path = saver.save(sess, 'DAC_models/' + model_name)
        print("Model saved in file: %s" % save_path)

      epoch += 1

    model_name = 'DAC_model_final_' + str(round(time()))+ '.ckpt'
    #save_path = saver.save(sess, 'DAC_models/' + model_name)
    save_path = saver.save(sess, '/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/DAC/res/' + model_name)

    print("Model saved in file: %s" % save_path)
    print("Total epochs: %d" % epoch)

    feed_dict_1={data_pool_input: train_data}
    pred_cluster = sess.run(pred_label, feed_dict=feed_dict_1)
    return pred_cluster

result = train()

result

from sklearn.metrics import silhouette_samples, silhouette_score
def silhouette_score1(X, labels, *, metric="euclidean", sample_size=None, random_state=None, **kwds):
 return np.mean(silhouette_samples(X, labels, metric=metric, **kwds))

def silhouette_score2(X, labels, *, metric="cosine", sample_size=None, random_state=None, **kwds):
 return np.mean(silhouette_samples(X, labels, metric=metric, **kwds))

silhouette_avg_rdata_daily = silhouette_score2(data_nor_eval, result)
print("The average silhouette_score is :", silhouette_avg_rdata_daily)

result

silhouette_avg_rdata_daily = silhouette_score2(data_nor_eval, result)
print("The average silhouette_score is :", silhouette_avg_rdata_daily)

result

u,indices = np.unique(result,return_counts = True) # sc=0.3412 st 64
u,indices

# silhouette_avg_rdata_daily_last = 0
# result_last = 0
# for ite in range(int(20)):
#   result = train()
#   silhouette_avg_rdata_daily = silhouette_score2(data_nor_eval, result)
#   print("The average silhouette_score is :", silhouette_avg_rdata_daily)
#   if silhouette_avg_rdata_daily_last < silhouette_avg_rdata_daily:
#     silhouette_avg_rdata_daily_last = silhouette_avg_rdata_daily
#     #print("The average silhouette_score is :", silhouette_avg_rdata_daily_last)
#     result_last = result
#     print("The result :", result_last)

def best_clustering5(n):

  n # Number of single models used
    #MIN_PROBABILITY = 0.6 # The minimum threshold of occurances of datapoints in a cluster

  sil = []
  sil_score = []
  # Generating a "Cluster Forest"
  #clustering_models = NUM_alg * [main()]#
  for i in range(n):
    result = train()
    #sil.append(results)
    #print(results)
    silhouette_avg_rdata_daily = silhouette_score(data_nor, result)
    sil_score.append(silhouette_avg_rdata_daily)

    sil.append([silhouette_avg_rdata_daily, result])

  print("Our silhouettes range is: ", sil_score)

  max_index = 0

  for j in range(len(sil)):
    if(sil[j][0]>=sil[max_index][0]):
      max_index = j
  print(sil[max_index])

  return sil[max_index]

def total_rmse(data_path,formed_clusters):
  # processed_data = data_preprocessing(data_path)
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
  return (print("The RMSE is:", rmse))

total_rmse('/content/drive/MyDrive/Data/ERA5_meteo_sfc_2021_daily.nc', result)

"""### This cell measure the variances of the generated clusters.  """

def avg_var(norm_data, clustering_results):
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

avg_var(data_nor, result)

"""### The following cell measure the average inter cluster distance.  """

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

avg_inter_dist(data_nor, result)

silhouette, result_1 = best_clustering5(5)

pickle.dump(result_1, open(path + 'DAC_hom_ens_' + str(silhouette) + '.pkl', "wb"))

silh = silhouette_score1(data_nor,  result_1)
u,indices = np.unique(result_1,return_counts = True) # sc=0.3412 st 64
# u,indices
print(silh)
print(u,indices)



silhouette, result_2 = best_clustering5(5)

pickle.dump(result_2, open(path + 'DAC_hom_ens_' + str(silhouette) + '.pkl', "wb"))

silh = silhouette_score1(data_nor,  result_2)
u,indices = np.unique(result_2,return_counts = True) # sc=0.3412 st 64
# u,indices
print(silh)
print(u,indices)

silhouette, result_3 = best_clustering5(5)

pickle.dump(result_3, open(path + 'DAC_hom_ens_' + str(silhouette) + '.pkl', "wb"))

silh = silhouette_score1(data_nor,  result_3)
u,indices = np.unique(result_3,return_counts = True) # sc=0.3412 st 64
# u,indices
print(silh)
print(u,indices)

silhouette, result_4 = best_clustering5(5)

pickle.dump(result_4, open(path + 'DAC_hom_ens_' + str(silhouette) + '.pkl', "wb"))

silh = silhouette_score1(data_nor,  result_4)
u,indices = np.unique(result_4,return_counts = True) # sc=0.3412 st 64
# u,indices
print(silh)
print(u,indices)

silhouette, result_5 = best_clustering5(5)

pickle.dump(result_5, open(path + 'DAC_hom_ens_' + str(silhouette) + '.pkl', "wb"))

silh = silhouette_score1(data_nor,  result_5)
u,indices = np.unique(result_5,return_counts = True) # sc=0.3412 st 64
# u,indices
print(silh)
print(u,indices)



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

clustering_models = NUM_alg * [result, result_1, result_2, result_3, result_4, result_5]


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

pickle.dump(final_labels, open(path + 'co-occ_ens_' + str(silh) + '.pkl', "wb"))

"""**Davies Bouldin**"""

