import sys
import numpy as np
import netCDF4
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import os
from netCDF4 import Dataset
import cartopy.io.shapereader as shapereader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.feature.nightshade import Nightshade
from PIL import Image
import pandas as pd
import numpy as np
import shutil
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from cartopy.feature import NaturalEarthFeature
import tensorflow as tf

def image_feature(path, direc):
    model = InceptionV3(weights='imagenet', include_top=False)
    features = [];
    img_name = [];
    for i in tqdm(direc):
        fname=path + 'train'+'/'+i
        print(fname)
        img=image.load_img(fname,target_size=(224,224))
        x = img_to_array(img)
        x=np.expand_dims(x,axis=0)
        x=preprocess_input(x)
        feat=model.predict(x)
        feat=feat.flatten()
        features.append(feat)
        img_name.append(i)
    return features,img_name

def image_saving(data_variable, saving_path):
        path = saving_path + "/" + data_variable._name
        print("creating path:" + path)
        os.makedirs(path, exist_ok=True)
        for index in range(data_variable.time.size):
                one_day_variable = data_variable.isel(time=index)
                fig = plt.figure(figsize=(161, 201))
                ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree()) #adding project within the opened figure
                mp = ax.imshow(one_day_variable-273.5,extent=(one_day_variable.longitude.min(),one_day_variable.longitude.max(), one_day_variable.latitude.min(), one_day_variable.latitude.max()),cmap='jet', origin='lower')
                plt.xlabel('x')
                plt.ylabel('y')

                #these are additional features you can add to geo plots (like boders, rivers, lakes..etc)
                states_provinces = cfeature.NaturalEarthFeature(
                        category='cultural',
                        name='admin_1_states_provinces_lines',
                        scale='10m',
                        facecolor='none')
                ax.add_feature(cfeature.BORDERS,edgecolor='blue')
                ax.add_feature(states_provinces, edgecolor='blue')
                ax.add_feature(cfeature.LAND)
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.OCEAN)
                ax.add_feature(cfeature.LAKES, alpha=0.5)
                ax.add_feature(cfeature.RIVERS)

                # adding colorbar and adjust the size
                # cbar = fig.colorbar(mp, shrink=0.4)
                # cbar.minorticks_on()

                #adding the long lat grids and enabling the tick labels
                gl = ax.gridlines(draw_labels=True,alpha=0.5)
                gl.top_labels = True
                gl.right_labels = True

                plt.savefig(path + "/" + data_variable._name + "_" + str(index) + ".jpg", dpi=10, bbox_inches='tight')
                plt.clf()
                plt.close(fig)


if __name__ == '__main__':
        data = xr.open_dataset("/Users/////Data/ECRP_ERA5/ERA5_sample_hourly_20200201-20200331.nc")
        print(data.data_vars)
        image_saving(data['v10'], "/Users/jianwu/Data/ECRP_ERA5/")
        #for data_key in data.data_vars:
        #        image_saving(data[data_key], "/Users/jianwu/Data/ECRP_ERA5/")


