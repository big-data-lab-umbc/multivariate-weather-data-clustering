
from google.colab import drive
drive.mount('/content/drive')

"""# **Multivariate Climate Data Clustering using Deep Temporal Clustering (DTC) Model**
Here we are dealing with climate data that comprises spatial information, time information, and scientific values. The dataset contains the value of 7 parameters for a region of 41 longitudes and 41 latitudes for 365 days in a year.

Our goal is to create meaningful clusters of 365 days based on the values of these 7 parameters. We have used the model suggested in the paper with the title "DEEP TEMPORAL CLUSTERING: FULLY UNSUPERVISED LEARNING OF TIME-DOMAIN FEATURES".

Source: https://github.com/FlorentF9/DeepTemporalClustering/blob/master/TAE.py

# **1. Model Creation:**
The authors used an autoencoder and a clustering layer in this model on top of the encoder output. DTC model takes a 2D array as input data. The model uses a three-level approach. The first level, implemented as a CNN, reduces the data dimensionality. The second level, implemented as a BI-LSTM, reduces the data dimensionality further and learns the temporal connections between time scales. The third level performs clustering of BI-LSTM latent representations. Mean square error (MSE) and KL divergence losses are used to train the model.
"""

# !pip install --upgrade pip

# !pip install --upgrade keras

# !pip install --upgrade tensorflow

!pip install tslearn==0.5.2

"""
These distance measures are used to compute the distance between two timesteps.
"""

import numpy as np
from statsmodels.tsa import stattools
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


def eucl(x, y):
    """
    Euclidean distance between two multivariate time series given as arrays of shape (timesteps, dim)
    """
    d = np.sqrt(np.sum(np.square(x - y), axis=0))
    return np.sum(d)


def cid(x, y):
    """
    Complexity-Invariant Distance (CID) between two multivariate time series given as arrays of shape (timesteps, dim)
    Reference: Batista, Wang & Keogh (2011). A Complexity-Invariant Distance Measure for Time Series. https://doi.org/10.1137/1.9781611972818.60
    """
    assert(len(x.shape) == 2 and x.shape == y.shape)  # time series must have same length and dimensionality
    ce_x = np.sqrt(np.sum(np.square(np.diff(x, axis=0)), axis=0) + 1e-9)
    ce_y = np.sqrt(np.sum(np.square(np.diff(y, axis=0)), axis=0) + 1e-9)
    d = np.sqrt(np.sum(np.square(x - y), axis=0)) * np.divide(np.maximum(ce_x, ce_y), np.minimum(ce_x, ce_y))
    return np.sum(d)


def cor(x, y):
    """
    Correlation-based distance (COR) between two multivariate time series given as arrays of shape (timesteps, dim)
    """
    scaler = TimeSeriesScalerMeanVariance()
    x_norm = scaler.fit_transform(x)
    y_norm = scaler.fit_transform(y)
    pcc = np.mean(x_norm * y_norm)  # Pearson correlation coefficients
    d = np.sqrt(2.0 * (1.0 - pcc + 1e-9))  # correlation-based similarities
    return np.sum(d)


def acf(x, y):
    """
    Autocorrelation-based distance (ACF) between two multivariate time series given as arrays of shape (timesteps, dim)
    Computes a linearly weighted euclidean distance between the autocorrelation coefficients of the input time series.
    Reference: Galeano & Pena (2000). Multivariate Analysis in Vector Time Series.
    """
    assert (len(x.shape) == 2 and x.shape == y.shape)  # time series must have same length and dimensionality
    x_acf = np.apply_along_axis(lambda z: stattools.acf(z, nlags=z.shape[0]), 0, x)
    y_acf = np.apply_along_axis(lambda z: stattools.acf(z, nlags=z.shape[0]), 0, y)
    weights = np.linspace(1.0, 0.0, x.shape[0])
    d = np.sqrt(np.sum(np.expand_dims(weights, axis=1) * np.square(x_acf - y_acf), axis=0))
    return np.sum(d)

"""
The clustering layer is defined here.
"""

#from keras.engine.topology import Layer, InputSpec
from tensorflow.keras.layers import Layer, InputSpec
import keras.backend as K


class TSClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, timesteps, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
        dist_metric: distance metric between sequences used in similarity kernel ('eucl', 'cir', 'cor' or 'acf').
    # Input shape
        3D tensor with shape: `(n_samples, timesteps, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, dist_metric='eucl', **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(TSClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.dist_metric = dist_metric
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=3)
        self.clusters = None
        self.built = False

    def build(self, input_shape):
        assert len(input_shape) == 3
        input_dim = input_shape[2]
        input_steps = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_steps, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_steps, input_dim), initializer='glorot_uniform', name='cluster_centers')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """
        Student t-distribution kernel, probability of assigning encoded sequence i to cluster k.
            q_{ik} = (1 + dist(z_i, m_k)^2)^{-1} / normalization.
        Arguments:
            inputs: encoded input sequences, shape=(n_samples, timesteps, n_features)
        Return:
            q: soft labels for each sample. shape=(n_samples, n_clusters)
        """
        if self.dist_metric == 'eucl':
            distance = K.sum(K.sqrt(K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2)), axis=-1)
        elif self.dist_metric == 'cid':
            ce_x = K.sqrt(K.sum(K.square(inputs[:, 1:, :] - inputs[:, :-1, :]), axis=1))  # shape (n_samples, n_features)
            ce_w = K.sqrt(K.sum(K.square(self.clusters[:, 1:, :] - self.clusters[:, :-1, :]), axis=1))  # shape (n_clusters, n_features)
            ce = K.maximum(K.expand_dims(ce_x, axis=1), ce_w) / K.minimum(K.expand_dims(ce_x, axis=1), ce_w)  # shape (n_samples, n_clusters, n_features)
            ed = K.sqrt(K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2))  # shape (n_samples, n_clusters, n_features)
            distance = K.sum(ed * ce, axis=-1)  # shape (n_samples, n_clusters)
        elif self.dist_metric == 'cor':
            inputs_norm = (inputs - K.expand_dims(K.mean(inputs, axis=1), axis=1)) / K.expand_dims(K.std(inputs, axis=1), axis=1)  # shape (n_samples, timesteps, n_features)
            clusters_norm = (self.clusters - K.expand_dims(K.mean(self.clusters, axis=1), axis=1)) / K.expand_dims(K.std(self.clusters, axis=1), axis=1)  # shape (n_clusters, timesteps, n_features)
            pcc = K.mean(K.expand_dims(inputs_norm, axis=1) * clusters_norm, axis=2)  # Pearson correlation coefficients
            distance = K.sum(K.sqrt(2.0 * (1.0 - pcc)), axis=-1)  # correlation-based similarities, shape (n_samples, n_clusters)
        elif self.dist_metric == 'acf':
            raise NotImplementedError
        else:
            raise ValueError('Available distances are eucl, cid, cor and acf!')
        q = 1.0 / (1.0 + K.square(distance) / self.alpha)
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 3
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters, 'dist_metric': self.dist_metric}
        base_config = super(TSClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

"""
The structure of the encoder and decoder is defined here. To define the autoencoder the encoder and decoder are merged together.
To train this model without GPU comment out the CuDNNLSTM layers and use LSTM layers.
"""

from keras.models import Model
#from keras.layers import Dropout,BatchNormalization, CuDNNLSTM
from keras.layers import Input, Conv1D, LeakyReLU, MaxPool1D, LSTM, Bidirectional, TimeDistributed, Dense, Reshape
# from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from keras.layers import UpSampling2D, Conv2DTranspose


def temporal_autoencoder(input_dim, timesteps, n_filters=50, kernel_size=10, strides=1, pool_size=10, n_units=[50, 1]):
    """
    Temporal Autoencoder (TAE) model with Convolutional and BiLSTM layers.
    # Arguments
        input_dim: input dimension
        timesteps: number of timesteps (can be None for variable length sequences)
        n_filters: number of filters in convolutional layer
        kernel_size: size of kernel in convolutional layer
        strides: strides in convolutional layer
        pool_size: pooling size in max pooling layer, must divide time series length
        n_units: numbers of units in the two BiLSTM layers
        alpha: coefficient in Student's kernel
        dist_metric: distance metric between latent sequences
    # Return
        (ae_model, encoder_model, decoder_model): AE, encoder and decoder models
    """
    assert(timesteps % pool_size == 0)

    # Input
    x = Input(shape=(timesteps, input_dim), name='input_seq')

    # Encoder
    encoded = Conv1D(n_filters, kernel_size, strides=strides, padding='same', activation='linear')(x)
    encoded = LeakyReLU()(encoded)
    encoded = MaxPool1D(pool_size)(encoded)
    encoded = Bidirectional(CuDNNLSTM(n_units[0], return_sequences=True), merge_mode='sum')(encoded)
    #encoded = Bidirectional(LSTM(n_units[0], return_sequences=True), merge_mode='sum')(encoded)
    encoded = LeakyReLU()(encoded)
    encoded = Bidirectional(CuDNNLSTM(n_units[1], return_sequences=True), merge_mode='sum')(encoded)
    #encoded = Bidirectional(LSTM(n_units[1], return_sequences=True), merge_mode='sum')(encoded)
    encoded = LeakyReLU(name='latent')(encoded)

    # Decoder
    decoded = Reshape((-1, 1, n_units[1]), name='reshape')(encoded)
    decoded = UpSampling2D((pool_size, 1), name='upsampling')(decoded)  #decoded = UpSampling1D(pool_size, name='upsampling')(decoded)
    decoded = Conv2DTranspose(input_dim, (kernel_size, 1), padding='same', name='conv2dtranspose')(decoded)
    output = Reshape((-1, input_dim), name='output_seq')(decoded)  #output = Conv1D(1, kernel_size, strides=strides, padding='same', activation='linear', name='output_seq')(decoded)

    # AE model
    autoencoder = Model(inputs=x, outputs=output, name='AE')

    # Encoder model
    encoder = Model(inputs=x, outputs=encoded, name='encoder')

    # Create input for decoder model
    encoded_input = Input(shape=(timesteps // pool_size, n_units[1]), name='decoder_input')

    # Internal layers in decoder
    decoded = autoencoder.get_layer('reshape')(encoded_input)
    decoded = autoencoder.get_layer('upsampling')(decoded)
    decoded = autoencoder.get_layer('conv2dtranspose')(decoded)
    decoder_output = autoencoder.get_layer('output_seq')(decoded)

    # Decoder model
    decoder = Model(inputs=encoded_input, outputs=decoder_output, name='decoder')

    return autoencoder, encoder, decoder

"""
The complete DTC model is created in this block using the autoencoder and clustering layer.
"""

# Utilities
import os
import csv
import argparse
from time import time

# Keras
from keras.models import Model
from keras.layers import Dense, Reshape, UpSampling2D, Conv2DTranspose, GlobalAveragePooling1D, Softmax
from keras.losses import kullback_leibler_divergence
import keras.backend as K

# scikit-learn
from sklearn.cluster import AgglomerativeClustering, KMeans

import numpy as np


class DTC:
    """
    Deep Temporal Clustering (DTC) model
    # Arguments
        n_clusters: number of clusters
        input_dim: input dimensionality
        timesteps: length of input sequences (can be None for variable length)
        n_filters: number of filters in convolutional layer
        kernel_size: size of kernel in convolutional layer
        strides: strides in convolutional layer
        pool_size: pooling size in max pooling layer, must divide the time series length
        n_units: numbers of units in the two BiLSTM layers
        alpha: coefficient in Student's kernel
        dist_metric: distance metric between latent sequences
        cluster_init: cluster initialization method
    """

    def __init__(self, n_clusters, input_dim, timesteps,
                 n_filters=50, kernel_size=10, strides=1, pool_size=8, n_units=[50, 1],
                 alpha=1.0, dist_metric='eucl', cluster_init='kmeans', heatmap=False):
        assert(timesteps % pool_size == 0)
        self.n_clusters = n_clusters
        self.input_dim = input_dim
        self.timesteps = timesteps
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.pool_size = pool_size
        self.n_units = n_units
        self.latent_shape = (self.timesteps // self.pool_size, self.n_units[1])
        self.alpha = alpha
        self.dist_metric = dist_metric
        self.cluster_init = cluster_init
        self.heatmap = heatmap
        self.pretrained = False
        self.model = self.autoencoder = self.encoder = self.decoder = None
        if self.heatmap:
            self.heatmap_model = None
            self.heatmap_loss_weight = None
            self.initial_heatmap_loss_weight = None
            self.final_heatmap_loss_weight = None
            self.finetune_heatmap_at_epoch = None

    def initialize(self):
        """
        Create DTC model
        """
        # Create AE models
        self.autoencoder, self.encoder, self.decoder = temporal_autoencoder(input_dim=self.input_dim,
                                                                            timesteps=self.timesteps,
                                                                            n_filters=self.n_filters,
                                                                            kernel_size=self.kernel_size,
                                                                            strides=self.strides,
                                                                            pool_size=self.pool_size,
                                                                            n_units=self.n_units)
        clustering_layer = TSClusteringLayer(self.n_clusters,
                                             alpha=self.alpha,
                                             dist_metric=self.dist_metric,
                                             name='TSClustering')(self.encoder.output)

        # Heatmap-generating network
        if self.heatmap:
            n_heatmap_filters = self.n_clusters  # one heatmap (class activation map) per cluster
            encoded = self.encoder.output
            heatmap_layer = Reshape((-1, 1, self.n_units[1]))(encoded)
            heatmap_layer = UpSampling2D((self.pool_size, 1))(heatmap_layer)
            heatmap_layer = Conv2DTranspose(n_heatmap_filters, (self.kernel_size, 1), padding='same')(heatmap_layer)
            # The next one is the heatmap layer we will visualize
            heatmap_layer = Reshape((-1, n_heatmap_filters), name='Heatmap')(heatmap_layer)
            heatmap_output_layer = GlobalAveragePooling1D()(heatmap_layer)
            # A dense layer must be added only if `n_heatmap_filters` is different from `n_clusters`
            # heatmap_output_layer = Dense(self.n_clusters, activation='relu')(heatmap_output_layer)
            heatmap_output_layer = Softmax()(heatmap_output_layer)  # normalize activations with softmax

        if self.heatmap:
            # Create DTC model
            self.model = Model(inputs=self.autoencoder.input,
                               outputs=[self.autoencoder.output, clustering_layer, heatmap_output_layer])
            # Create Heatmap model
            self.heatmap_model = Model(inputs=self.autoencoder.input,
                                       outputs=heatmap_layer)
        else:
            # Create DTC model
            self.model = Model(inputs=self.autoencoder.input,
                               outputs=[self.autoencoder.output, clustering_layer])

    @property
    def cluster_centers_(self):
        """
        Returns cluster centers
        """
        return self.model.get_layer(name='TSClustering').get_weights()[0]

    @staticmethod
    def weighted_kld(loss_weight):
        """
        Custom KL-divergence loss with a variable weight parameter
        """
        def loss(y_true, y_pred):
            return loss_weight * kullback_leibler_divergence(y_true, y_pred)
        return loss

    def on_epoch_end(self, epoch):
        """
        Update heatmap loss weight on epoch end
        """
        if epoch > self.finetune_heatmap_at_epoch:
            K.set_value(self.heatmap_loss_weight, self.final_heatmap_loss_weight)

    def compile(self, gamma, optimizer, initial_heatmap_loss_weight=None, final_heatmap_loss_weight=None):
        """
        Compile DTC model
        # Arguments
            gamma: coefficient of TS clustering loss
            optimizer: optimization algorithm
            initial_heatmap_loss_weight (optional): initial weight of heatmap loss vs clustering loss
            final_heatmap_loss_weight (optional): final weight of heatmap loss vs clustering loss (heatmap finetuning)
        """
        if self.heatmap:
            self.initial_heatmap_loss_weight = initial_heatmap_loss_weight
            self.final_heatmap_loss_weight = final_heatmap_loss_weight
            self.heatmap_loss_weight = K.variable(self.initial_heatmap_loss_weight)
            self.model.compile(loss=['mse', DTC.weighted_kld(1.0 - self.heatmap_loss_weight), DTC.weighted_kld(self.heatmap_loss_weight)],
                               loss_weights=[1.0, gamma, gamma],
                               optimizer=optimizer)
        else:
            self.model.compile(loss=['mse', 'kld'],
                               loss_weights=[1.0, gamma],
                               optimizer=optimizer)

    def load_weights(self, weights_path):
        """
        Load pre-trained weights of DTC model
        # Arguments
            weight_path: path to weights file (.h5)
        """
        self.model.load_weights(weights_path)
        self.pretrained = True

    def load_ae_weights(self, ae_weights_path):
        """
        Load pre-trained weights of AE
        # Arguments
            ae_weight_path: path to weights file (.h5)
        """
        self.autoencoder.load_weights(ae_weights_path)
        self.pretrained = True

    def dist(self, x1, x2):
        """
        Compute distance between two multivariate time series using chosen distance metric
        # Arguments
            x1: first input (np array)
            x2: second input (np array)
        # Return
            distance
        """
        if self.dist_metric == 'eucl':
            return eucl(x1, x2)
        elif self.dist_metric == 'cid':
            return cid(x1, x2)
        elif self.dist_metric == 'cor':
            return cor(x1, x2)
        elif self.dist_metric == 'acf':
            return acf(x1, x2)
        else:
            raise ValueError('Available distances are eucl, cid, cor and acf!')

    def init_cluster_weights(self, X):
        """
        Initialize with complete-linkage hierarchical clustering or k-means.
        # Arguments
            X: numpy array containing training set or batch
        """
        assert(self.cluster_init in ['hierarchical', 'kmeans'])
        print('Initializing cluster...')

        features = self.encode(X)

        if self.cluster_init == 'hierarchical':
            if self.dist_metric == 'eucl':  # use AgglomerativeClustering off-the-shelf
                hc = AgglomerativeClustering(n_clusters=self.n_clusters,
                                             affinity='euclidean',
                                             linkage='complete').fit(features.reshape(features.shape[0], -1))
            else:  # compute distance matrix using dist
                d = np.zeros((features.shape[0], features.shape[0]))
                for i in range(features.shape[0]):
                    for j in range(i):
                        d[i, j] = d[j, i] = self.dist(features[i], features[j])
                hc = AgglomerativeClustering(n_clusters=self.n_clusters,
                                             affinity='precomputed',
                                             linkage='complete').fit(d)
            # compute centroid
            cluster_centers = np.array([features[hc.labels_ == c].mean(axis=0) for c in range(self.n_clusters)])
        elif self.cluster_init == 'kmeans':
            # fit k-means on flattened features
            km = KMeans(n_clusters=self.n_clusters, n_init=10).fit(features.reshape(features.shape[0], -1))
            cluster_centers = km.cluster_centers_.reshape(self.n_clusters, features.shape[1], features.shape[2])

        self.model.get_layer(name='TSClustering').set_weights([cluster_centers])
        print('Done!')

    def encode(self, x):
        """
        Encoding function. Extract latent features from hidden layer
        # Arguments
            x: data point
        # Return
            encoded (latent) data point
        """
        return self.encoder.predict(x)

    def decode(self, x):
        """
        Decoding function. Decodes encoded sequence from latent space.
        # Arguments
            x: encoded (latent) data point
        # Return
            decoded data point
        """
        return self.decoder.predict(x)

    def predict(self, x):
        """
        Predict cluster assignment.
        """
        q = self.model.predict(x, verbose=0)[1]
        return q.argmax(axis=1)

    @staticmethod
    def target_distribution(q):  # target distribution p which enhances the discrimination of soft label q
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def predict_heatmap(self, x):
        """
        Produces TS clustering heatmap from input sequence.
        # Arguments
            x: data point
        # Return
            heatmap
        """
        return self.heatmap_model.predict(x, verbose=0)

    def pretrain(self, X,
                 optimizer='adam',
                 epochs=10,
                 batch_size=64,
                 save_dir='/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/DTC-Clustering-Model',
                 verbose=1):
        """
        Pre-train the autoencoder using only MSE reconstruction loss
        Saves weights in h5 format.
        # Arguments
            X: training set
            optimizer: optimization algorithm
            epochs: number of pre-training epochs
            batch_size: training batch size
            save_dir: path to existing directory where weights will be saved
        """
        print('Pretraining...')
        self.autoencoder.compile(optimizer=optimizer, loss='mse')

        # Begin pretraining
        t0 = time()
        self.autoencoder.fit(X, X, batch_size=batch_size, epochs=epochs, verbose=verbose)
        print('Pretraining time: ', time() - t0)
        self.autoencoder.save_weights('{}/ae_weights-epoch{}.h5'.format(save_dir, epochs))
        print('Pretrained weights are saved to {}/ae_weights-epoch{}.h5'.format(save_dir, epochs))
        self.pretrained = True

    def fit(self, X_train, y_train=None,
            X_val=None, y_val=None,
            epochs=100,
            eval_epochs=10,
            save_epochs=10,
            batch_size=64,
            tol=0.001,
            patience=5,
            finetune_heatmap_at_epoch=8,
            save_dir='/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/DTC-Clustering-Model'):
        """
        Training procedure
        # Arguments
           X_train: training set
           y_train: (optional) training labels
           X_val: (optional) validation set
           y_val: (optional) validation labels
           epochs: number of training epochs
           eval_epochs: evaluate metrics on train/val set every eval_epochs epochs
           save_epochs: save model weights every save_epochs epochs
           batch_size: training batch size
           tol: tolerance for stopping criterion
           patience: patience for stopping criterion
           finetune_heatmap_at_epoch: epoch number where heatmap finetuning will start. Heatmap loss weight will
                                      switch from `self.initial_heatmap_loss_weight` to `self.final_heatmap_loss_weight`
           save_dir: path to existing directory where weights and logs are saved
        """
        if not self.pretrained:
            print('Autoencoder was not pre-trained!')

        if self.heatmap:
            self.finetune_heatmap_at_epoch = finetune_heatmap_at_epoch

        # Logging file
        logfile = open(save_dir + '/dtc_log.csv', 'w')
        fieldnames = ['epoch', 'T', 'L', 'Lr', 'Lc']
        if X_val is not None:
            fieldnames += ['L_val', 'Lr_val', 'Lc_val']
        if y_train is not None:
            fieldnames += ['acc', 'pur', 'nmi', 'ari']
        if y_val is not None:
            fieldnames += ['acc_val', 'pur_val', 'nmi_val', 'ari_val']
        logwriter = csv.DictWriter(logfile, fieldnames)
        logwriter.writeheader()

        y_pred_last = None
        patience_cnt = 0

        print('Training for {} epochs.\nEvaluating every {} and saving model every {} epochs.'.format(epochs, eval_epochs, save_epochs))

        for epoch in range(epochs):

            # Compute cluster assignments for training set
            q = self.model.predict(X_train)[1]
            p = DTC.target_distribution(q)

            # Evaluate losses and metrics on training set
            if epoch % eval_epochs == 0:

                # Initialize log dictionary
                logdict = dict(epoch=epoch)

                y_pred = q.argmax(axis=1)
                if X_val is not None:
                    q_val = self.model.predict(X_val)[1]
                    p_val = DTC.target_distribution(q_val)
                    y_val_pred = q_val.argmax(axis=1)

                print('epoch {}'.format(epoch))
                if self.heatmap:
                    loss = self.model.evaluate(X_train, [X_train, p, p], batch_size=batch_size, verbose=False)
                else:
                    loss = self.model.evaluate(X_train, [X_train, p], batch_size=batch_size, verbose=False)
                logdict['L'] = loss[0]
                logdict['Lr'] = loss[1]
                logdict['Lc'] = loss[2]
                print('[Train] - Lr={:f}, Lc={:f} - total loss={:f}'.format(logdict['Lr'], logdict['Lc'], logdict['L']))
                if X_val is not None:
                    val_loss = self.model.evaluate(X_val, [X_val, p_val], batch_size=batch_size, verbose=False)
                    logdict['L_val'] = val_loss[0]
                    logdict['Lr_val'] = val_loss[1]
                    logdict['Lc_val'] = val_loss[2]
                    print('[Val] - Lr={:f}, Lc={:f} - total loss={:f}'.format(logdict['Lr_val'], logdict['Lc_val'], logdict['L_val']))

                # Evaluate the clustering performance using labels
                if y_train is not None:
                    logdict['acc'] = cluster_acc(y_train, y_pred)
                    logdict['pur'] = cluster_purity(y_train, y_pred)
                    logdict['nmi'] = metrics.normalized_mutual_info_score(y_train, y_pred)
                    logdict['ari'] = metrics.adjusted_rand_score(y_train, y_pred)
                    print('[Train] - Acc={:f}, Pur={:f}, NMI={:f}, ARI={:f}'.format(logdict['acc'], logdict['pur'],
                                                                                    logdict['nmi'], logdict['ari']))
                if y_val is not None:
                    logdict['acc_val'] = cluster_acc(y_val, y_val_pred)
                    logdict['pur_val'] = cluster_purity(y_val, y_val_pred)
                    logdict['nmi_val'] = metrics.normalized_mutual_info_score(y_val, y_val_pred)
                    logdict['ari_val'] = metrics.adjusted_rand_score(y_val, y_val_pred)
                    print('[Val] - Acc={:f}, Pur={:f}, NMI={:f}, ARI={:f}'.format(logdict['acc_val'], logdict['pur_val'],
                                                                                  logdict['nmi_val'], logdict['ari_val']))

                logwriter.writerow(logdict)

                # check stop criterion
                if y_pred_last is not None:
                    assignment_changes = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred
                if epoch > 0 and assignment_changes < tol:
                    patience_cnt += 1
                    print('Assignment changes {} < {} tolerance threshold. Patience: {}/{}.'.format(assignment_changes, tol, patience_cnt, patience))
                    if patience_cnt >= patience:
                        print('Reached max patience. Stopping training.')
                        logfile.close()
                        break
                else:
                    patience_cnt = 0

            # Save intermediate model and plots
            if epoch % save_epochs == 0:
                self.model.save_weights(save_dir + '/DTC_model_' + str(epoch) + '.h5')
                print('Saved model to:', save_dir + '/DTC_model_' + str(epoch) + '.h5')

            # Train for one epoch
            if self.heatmap:
                self.model.fit(X_train, [X_train, p, p], epochs=1, batch_size=batch_size, verbose=False)
                self.on_epoch_end(epoch)
            else:
                self.model.fit(X_train, [X_train, p], epochs=1, batch_size=batch_size, verbose=False)

        # Save the final model
        logfile.close()
        print('Saving model to:', save_dir + '/DTC_model_final.h5')
        self.model.save_weights(save_dir + '/DTC_model_final.h5')

!pip install netCDF4

#!pip3 install --upgrade tensorflow

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
from time import time
warnings.filterwarnings("ignore")

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

path = '/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/'

data_path = ('/content/drive/MyDrive/Data/ERA5_meteo_sfc_2021_daily.nc')
nor_data = data_preprocessing(data_path)

nor_data.shape

data_nor = nor_data

from sklearn.metrics import silhouette_samples, silhouette_score
def silhouette_score1(X, labels, *, metric="cosine", sample_size=None, random_state=None, **kwds):
 return np.mean(silhouette_samples(X, labels, metric=metric, **kwds))

"""## **3. Model Training**
This function defines related parameters to train the model. Then instantiate the model and train on the pre-processed data. The model tries to optimize the clustering loss. After training the model returns the cluster results.
"""

class Config(object):
    ae_weights = None #, help='pre-trained autoencoder weights')
    n_clusters = 7 #, type=int, help='number of clusters')
    n_filters = 50 #, type=int, help='number of filters in convolutional layer')
    kernel_size = 10 #,  type=int, help='size of kernel in convolutional layer')
    strides = 1 #, type=int, help='strides in convolutional layer')
    pool_size = 8 #, type=int, help='pooling size in max pooling layer')
    n_units = [50, 1] #, type=int, help='numbers of units in the BiLSTM layers')
    gamma = 1.0 #, type=float, help='coefficient of clustering loss')
    alpha = 1.0 #, type=float, help='coefficient in Student\'s kernel')
    dist_metric ='eucl' #, type=str, choices=['eucl', 'cid', 'cor', 'acf'], help='distance metric between latent sequences')
    cluster_init = 'kmeans' #, type=str, choices=['kmeans', 'hierarchical'], help='cluster initialization method')
    heatmap = False #, type=bool, help='train heatmap-generating network')
    pretrain_epochs = 10 #, type=int)
    epochs = 100 #, type=int)
    eval_epochs = 1 #, type=int)
    save_epochs = 50 #, type=int)
    batch_size = 366 #, type=int)
    tol = 0.00001 #, type=float, help='tolerance for stopping criterion')
    patience = 10 #, type=int, help='patience for stopping criterion')
    finetune_heatmap_at_epoch = 8 #, type=int, help='epoch where heatmap finetuning starts')
    initial_heatmap_loss_weight = 0.1 #, type=float, help='initial weight of heatmap loss vs clustering loss')
    final_heatmap_loss_weight = 0.9 #, type=float, help='final weight of heatmap loss vs clustering loss (heatmap finetuning)')
    save_dir = '/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/DTC-Clustering-Model'




def main():

    args = Config()
    print(args)

    # Create save directory if not exists
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Load data
    data_path = '/content/drive/MyDrive/Data/ERA5_meteo_sfc_2021_daily.nc'
    xt = data_preprocessing(data_path)
    xt = np.append(xt,np.zeros([len(xt),1]),1)
    X_train = np.reshape(xt, xt.shape + (1,))
    y_train = None
    X_val = None
    y_val = None

    # Find number of clusters
    if args.n_clusters is None:
        args.n_clusters = 7 #len(np.unique(y_train))

    # Set default values
    pretrain_optimizer = 'adam'
    print("Timestep: ", X_train.shape[1], "Pool Size: ", args.pool_size)

    # Instantiate model
    dtc = DTC(n_clusters=args.n_clusters,
              input_dim=X_train.shape[-1],
              timesteps=X_train.shape[1],
              n_filters=args.n_filters,
              kernel_size=args.kernel_size,
              strides=args.strides,
              pool_size=args.pool_size,
              n_units=args.n_units,
              alpha=args.alpha,
              dist_metric=args.dist_metric,
              cluster_init=args.cluster_init,
              heatmap=args.heatmap)

    # Initialize model
    optimizer = 'adam'
    dtc.initialize()
    dtc.model.summary()
    dtc.compile(gamma=args.gamma, optimizer=optimizer, initial_heatmap_loss_weight=args.initial_heatmap_loss_weight,
                final_heatmap_loss_weight=args.final_heatmap_loss_weight)

    # Load pre-trained AE weights or pre-train
    if args.ae_weights is None and args.pretrain_epochs > 0:
        dtc.pretrain(X=X_train, optimizer=pretrain_optimizer,
                     epochs=args.pretrain_epochs, batch_size=args.batch_size,
                     save_dir=args.save_dir)
    elif args.ae_weights is not None:
        dtc.load_ae_weights(args.ae_weights)

    # Initialize clusters
    dtc.init_cluster_weights(X_train)

    # Fit model
    t0 = time()
    dtc.fit(X_train, y_train, X_val, y_val, args.epochs, args.eval_epochs, args.save_epochs, args.batch_size,
            args.tol, args.patience, args.finetune_heatmap_at_epoch, args.save_dir)
    print('Training time: ', (time() - t0))

    # Evaluate
    print('Performance (TRAIN)')
    results = {}
    q = dtc.model.predict(X_train)[1]
    y_pred = q.argmax(axis=1)
    return y_pred

# result = main()
# result



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

silh = silhouette_score1(data_nor,  results)
u,indices = np.unique(results,return_counts = True) # sc=0.3412 st 64
# u,indices
print(silh)
print(u,indices)

from sklearn.metrics import davies_bouldin_score
print("Davies-Bouldin score is ", davies_bouldin_score(data_nor, results))

from sklearn.metrics import calinski_harabasz_score
ch_ind = calinski_harabasz_score(data_nor, results)
ch_ind

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

pickle.dump(result_2, open(path + 'DTC_hom_ens_' + str(silhouette) + '.pkl', "wb"))

silh = silhouette_score1(data_nor,  result_2)
u,indices = np.unique(result_2,return_counts = True) # sc=0.3412 st 64
# u,indices
print(silh)
print(u,indices)

silhouette, result_3 = best_clustering(20)

pickle.dump(result_3, open(path + 'DTC_hom_ens_' + str(silhouette) + '.pkl', "wb"))

data_nor_eval = data_nor

silh = silhouette_score1(data_nor,  result_3)
u,indices = np.unique(result_3,return_counts = True) # sc=0.3412 st 64
# u,indices
print(silh)
print(u,indices)

silhouette, result_4 = best_clustering(20)

pickle.dump(result_4, open(path + 'DTC_hom_ens_' + str(silhouette) + '.pkl', "wb"))

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

pickle.dump(result_5, open(path + 'DTC_hom_ens_' + str(silhouette) + '.pkl', "wb"))

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
np.save('/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/DSC_ens_co_occurrence_matrix.npy', sim_matrixx)
#print(norm_sim_matrix)

import numpy as geek

data_nor_eval = data_nor

#sim_matrixx = geek.load('/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Non-negative Matrix Factorization/DEC_co_occurrence_matrix.npy')

from sklearn.cluster import SpectralClustering
spec_clt = SpectralClustering(n_clusters=7, affinity='precomputed',
                              n_init=5, random_state=214)
final_labels = spec_clt.fit_predict(sim_matrixx)

pickle.dump(final_labels, open(path + 'DSC_fin_ens_' + str(silh) + '.pkl', "wb"))

# result = pickle.load(open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/DSC_Ensemble1_032.pkl", "rb"))

"""# **Evaluation Metrics**"""

silh = silhouette_score1(data_nor,  final_labels)
u,indices = np.unique(final_labels,return_counts = True)

print("Silhouette score is ", silh)
print("Cluster index ", u, "and Cluster Sizes: ", indices)

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



# result = pickle.load(open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/DTC_295.pkl", "rb"))
# result_1 = pickle.load(open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/DTC_295.pkl", "rb"))
# result_2 = pickle.load(open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/DTC_3367.pkl", "rb"))
# result_3 = pickle.load(open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/DTC_3124.pkl", "rb"))
# result_4 = pickle.load(open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/DTC_3226.pkl", "rb"))
# result_5 = pickle.load(open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/DTC_02014.pkl", "rb"))

clusters_list = [ result_1, result_2, result_3, result_4,result_5]



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

clustering_models = NUM_alg * [result_1, result_2, result_3,result_4,result_5]

clt_sim_matrix = ClusterSimilarityMatrix() #Initializing the similarity matrix
for model in clustering_models:
  clt_sim_matrix.fit(model)

sim_matrixx = clt_sim_matrix.similarity
sim_matrixx = sim_matrixx/sim_matrixx.diagonal()
sim_matrixx[sim_matrixx < occurrence_threshold] = 0

# norm_sim_matrix[norm_sim_matrix < occurrence_threshold] = 0

unique_labels = np.unique(np.concatenate(clustering_models))

print(sim_matrixx)
np.save('/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Non-negative Matrix Factorization/DTC_ens_co_occurrence_matrix.npy', sim_matrixx)
#print(norm_sim_matrix)

import numpy as geek

data_nor_eval = data_nor

#sim_matrixx = geek.load('/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Non-negative Matrix Factorization/DEC_co_occurrence_matrix.npy')

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

def best_clustering5(n):

  n # Number of single models used
    #MIN_PROBABILITY = 0.6 # The minimum threshold of occurances of datapoints in a cluster

  sil = []
  sil_score = []
  # Generating a "Cluster Forest"
  #clustering_models = NUM_alg * [main()]#
  for i in range(n):
    spec_clt = SpectralClustering(n_clusters=7, affinity='precomputed',
                              n_init=5, random_state=214)

    results = spec_clt.fit_predict(sim_matrixx)

    silhouette_avg_rdata_daily = silhouette_score1(data_nor_eval, results)
    sil_score.append(silhouette_avg_rdata_daily)

    sil.append([silhouette_avg_rdata_daily, results])

  print("Our silhouettes range is: ", sil_score)

  max_index = 0

  for j in range(len(sil)):
    if(sil[j][0]>=sil[max_index][0]):
      max_index = j
  print(sil[max_index])

  #return sil[max_index]
  return sil[max_index]

# silhouette, result = best_clustering5(20)

silhouette, result_ek = best_clustering5(20)

silh = silhouette_score1(data_nor_eval,  result_ek)
u,indices = np.unique(result_ek,return_counts = True) # sc=0.3412 st 64
# u,indices
print(silh)
print(u,indices)



pickle.dump(final_labels, open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/DTC_hom_ens_034.pkl", "wb"))

#cluster_result = pickle.load(open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/test_clustering_results30.pkl", "rb"))



"""**Davies Bouldin**"""

from sklearn.metrics import davies_bouldin_score

print("Davies-Bouldin score is ", davies_bouldin_score(data_nor, result_ek))

"""#RMSE#"""



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
  return rmse

total_rmse('/content/drive/MyDrive/Data/ERA5_meteo_sfc_2021_daily.nc', result_ek)

"""### This cell measure the variances of the generated clusters.  """

trans_data = pd.DataFrame(data_nor)
trans_data['Cluster'] = result_ek
Clusters = {}
Cluster_Centers = {}
for i in set(result_ek):
  Clusters['Cluster' + str(i)] = np.array(trans_data[trans_data.Cluster == i].drop(columns=['Cluster']))

variances = pd.DataFrame(columns=range(len(Clusters)),index=range(2))
for i in range(len(Clusters)):
    variances[i].iloc[0] = np.var(Clusters['Cluster' + str(i)])
    variances[i].iloc[1] = Clusters['Cluster' + str(i)].shape[0]

var_sum = 0
for i in range(7):
    var_sum = var_sum + (variances[i].iloc[0] * variances[i].iloc[1])

var_avg = var_sum/data_nor.shape[0]
var_avg

"""### The following cell measure the average inter cluster distance.  """

from scipy.spatial.distance import cdist,pdist
n_clusters = 7
trans_data = pd.DataFrame(data_nor)
trans_data['Cluster'] = result_ek
Clusters = {}
Cluster_Centers = {}
for i in set(result_ek):
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
avg_inter





"""# **Different Approach**"""



cls_tup_list = []
for cls_tup in zip(*clusters_list):
    cls_tup_list.append(cls_tup)
zipper = {x: i for i, x in enumerate(sorted(set(cls_tup_list)))}
zipped_list = [zipper[x] for x in cls_tup_list]
unzipper = defaultdict(set)
for idx, cls_tup in enumerate(cls_tup_list):
    zipped = zipper[cls_tup]
    unzipper[zipped].add(idx)
comp_clusters_list = [[-1]*len(zipper) for _ in range(len(clusters_list))]
for clusters, comp_clusters in zip(clusters_list, comp_clusters_list):
    for i, cluster_i in enumerate(clusters):
            value = zipped_list[i]
            comp_clusters[value] = cluster_i

from tqdm import trange

def create_sparse_matrix(clusters):
    n = len(clusters)
    data = []
    row = []
    col = []
    # O(n**2)
    for i in trange(n):
        for j in range(i+1, n):
            if clusters[i] == clusters[j]:
                data.append(1)
                row.append(i)
                col.append(j)
    return csr_matrix((data, (row, col)), shape=(n, n))

sparse_matrix_list = [create_sparse_matrix(comp_clusters) for comp_clusters in comp_clusters_list]

sparse_matrix_mean = (sparse_matrix_list[0] * 0.5 + sparse_matrix_list[1] * 0.5)

# The lower the threshold the lower number of clusters: Since we are counting by the connected component.
# It seems desirable to set it a little higher.
threshold = 0.5
sparse_matrix_mean[sparse_matrix_mean < threshold] = 0

sparse_matrix_mean = sparse_matrix_mean.toarray()
sparse_matrix_mean

# Sort sparse_matrix_mean[node1][node2] that are non-zero.
# It is fast enough because the number of edges is small due to preprocessing (zipper).
clusters_final = np.zeros(len(data_nor))
clusters_final_next_id = 0

node_end = len(comp_clusters_list[0])
edge_list = []  # [(w, fr, to), ...]
for fr in range(node_end):
    for to in range(fr, node_end):
        w = sparse_matrix_mean[fr][to]
        if w == 0:
            continue
        edge_list.append((w, fr, to))
edge_list.sort(reverse=True)

SIZ_MAX = 18000
CLS_NUM_MIN = 7

class DSU:
    def __init__(self, node_end, unzipper):
        self.par = [i for i in range(node_end)]
        self.siz = [len(unzipper[i]) for i in range(node_end)]
        self.cls_num = node_end  # To use the cls_num variable, we declared this class

    def find(self, x):
        if self.par[x] == x: return x
        self.par[x] = self.find(self.par[x])
        return self.par[x]

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)
        if x == y:
            return
        if self.siz[x] > self.siz[y]: x, y = y, x
        self.par[x] = y
        self.siz[y] += self.siz[x]
        self.cls_num -= 1

    def get_siz(self, x):
        x = self.find(x)
        return self.siz[x]

dsu = DSU(node_end, unzipper)
for w, fr, to in edge_list:
    if (dsu.get_siz(fr)+dsu.get_siz(to)) > SIZ_MAX: continue
    dsu.union(fr, to)
    if dsu.cls_num <= CLS_NUM_MIN:
        print("number of clusters reaches CLS_NUM_MIN: {}".format(CLS_NUM_MIN), " break.")
        break

clusters_final = [0]*len(clusters_list[0])
for node in range(node_end):
    cluster_id = dsu.find(node)
    idx_list = unzipper[node]
    for idx in idx_list:
        clusters_final[idx] = cluster_id

zipper = {x: i for i, x in enumerate(sorted(set(clusters_final)))}
clusters_final = [zipper[x] for x in clusters_final]



df = pd.read_csv("/content/drive/MyDrive/ECRP_Data_Science/Implementation and Results/Jianwu_test/PCA_Result_Combination/PCA_Combined_var1.csv")

df2 = df.time_step
df2

df1 = data_nor
# creating pandas dataframe on transformed data
df1 = pd.DataFrame(df2, columns=['time_step'])
df1['clusterid'] =  clusters
#df1["cluster"] = cluster.labels_
df1['clusterid'].value_counts()
df1.to_csv("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/DEC_clustering.csv")
df1

df1.groupby('clusterid').count()



ensemble = silhouette_score1(data_nor_eval,  clusters)
u,indices = np.unique(clusters,return_counts = True) # sc=0.3412 st 64
#u,indices
print(ensemble)
print(u,indices)

import pickle

pickle.dump(clusters, open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/DSC_ensemble2_028.pkl", "wb"))

#cluster_result = pickle.load(open("/content/drive/MyDrive///Ensemble_Clustering/final/ensemble_alg/Final_ensemble/DSC_ensemble2_028.pkl", "rb"))

"""**Evaluations**"""

data_nor_eval = data_nor

result = clusters

"""**Silhouette**"""

silhouette_avg_rdata_daily = silhouette_score1(data_nor, result)
print("The average silhouette_score is :", silhouette_avg_rdata_daily)

"""**Davies Bouldin**"""

from sklearn.metrics import davies_bouldin_score

print("Davies-Bouldin score is ", davies_bouldin_score(data_nor, result))

"""#RMSE#"""



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
  return rmse

total_rmse('/content/drive/MyDrive/Data/ERA5_meteo_sfc_2021_daily.nc', result)

"""### This cell measure the variances of the generated clusters.  """

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
var_avg

"""### The following cell measure the average inter cluster distance.  """

from scipy.spatial.distance import cdist,pdist

trans_data = pd.DataFrame(data_nor)
trans_data['Cluster'] = result
Clusters = {}
Cluster_Centers = {}
for i in set(result):
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
avg_inter



