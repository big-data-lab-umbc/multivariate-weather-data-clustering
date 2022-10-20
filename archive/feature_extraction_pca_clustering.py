# -*- coding: utf-8 -*-
"""Feature_Extraction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Up6IExwHy3jfzjwfLaUEyKXEGBI5UMUT
"""

import sys
import numpy as np
import netCDF4
import xarray as xr
import matplotlib.pyplot as plt
import os
from PIL import Image
import pandas as pd
from netCDF4 import Dataset
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.cluster import KMeans
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA

# Function to Extract features from the images
def image_feature(path, image_size):
    direc = os.listdir(path)
    model = InceptionV3(weights='imagenet', include_top=False)
    features = [];
    img_names = [];
    for i in tqdm(direc):
        fname=path + '/' +i
        print(fname)
        img=image.load_img(fname,image_size)
        x = img_to_array(img)
        x=np.expand_dims(x,axis=0)
        x=preprocess_input(x)
        feat=model.predict(x)
        feat=feat.flatten()
        features.append(feat)
        img_names.append(i)
    return features,img_names

if __name__ == '__main__':
    var_name = "u10"
    img_path = "/Volumes/GoogleDrive/.shortcut-targets-by-id/1vfQuEpjPQbXwHTxqAw34ALMoA45PJ7KQ/ECRP_Data_Science/Zheng/new_data_images/" + var_name
    work_dir = "/Users/jianwu/Data/ECRP_ERA5/version-2/csv"
    # the size of the new figure is 41x41
    img_features, img_names = image_feature(img_path, (41, 41))

    #save original features
    img_features_df = pd.DataFrame(img_features)
    img_features_df['file_name'] = img_names
    csv_path = work_dir + "/" + var_name + ".csv"
    img_features_df.to_csv(csv_path, index=False)

    #clustering based original features
    k = 3
    clusters = KMeans(k, random_state = 40)
    clusters.fit(img_features)
    image_cluster = pd.DataFrame(img_names, columns=['image'])
    image_cluster["clusterid"] = clusters.labels_
    clustering_path = work_dir + "/" + var_name + "_clusters.csv"
    image_cluster.to_csv(clustering_path, index=False)

    #conduct PCA and save its results
    img_features_df = pd.DataFrame(img_features)
    print(img_features_df.shape)
    pca = PCA(3)
    img_reduced_features = pca.fit_transform(img_features_df)
    img_reduced_features_df = pd.DataFrame(img_reduced_features)
    img_reduced_features_df['file_name'] = img_names
    pca_csv_path = work_dir + "/" + var_name + "_pca.csv"
    img_reduced_features_df.to_csv(pca_csv_path, index=False)

    #clustering based reduced features
    k = 3
    clusters = KMeans(k, random_state = 40)
    clusters.fit(img_reduced_features)
    image_cluster = pd.DataFrame(img_names, columns=['image'])
    image_cluster["clusterid"] = clusters.labels_
    image_cluster
    pca_clustering_path = work_dir + "/" + var_name + "_pca_clusters.csv"
    image_cluster.to_csv(pca_clustering_path, index=False)

    print("done!")