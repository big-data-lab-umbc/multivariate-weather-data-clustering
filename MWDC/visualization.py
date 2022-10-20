import sys 
import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt 
from netCDF4 import Dataset
import matplotlib as mpl 
import matplotlib.colors as colors
import os

## read the timestep and it's cluster ID from your clustering results
def read_combined_cluster(csvlink,varid):
    with open(csvlink, mode ='r')as file:
       # reading the CSV file
       csvFile = pd.read_csv(file)
       if(len(varid)>0):    
          id = csvFile['clusterid']
          time_step = csvFile['time_step']
          #days = np.zeros(len(id))-999 
          days=np.arange(len(time_step))
          #for i in range(len(id)): 
          #    print()  
          #    days[i] = int(time_step[i][5:])
                            
                            
    return days,id


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


##called by plot_map to plot the coastline on the map
def plotcoastline(path, color='k'):
    lon_c = []
    lat_c = []
    with open(path) as f:
        for line in f:
            data = line.split()
            lon_c.append(float(data[0])-360)
            lat_c.append(float(data[1]))
    plt.plot(lon_c,lat_c,color=color,marker='s',markerfacecolor=color)
    return [lon_c,lat_c]