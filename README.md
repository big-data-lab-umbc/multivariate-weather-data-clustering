# multivariate-weather-data-clustering

## Installation

To install the package you need to create an environment using [pip](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) or [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). After that just clone this repository and install the ` setup.py` file inside it.

```bash
 git clone https://github.com/big-data-lab-umbc/multivariate-weather-data-clustering.git
 cd multivariate-weather-data-clustering
 python setup.py install
```

Note: If you are using macOS, you should use ` python3 setup.py install` instead.

## Usage

To use the functions you just need to import them from MWDC. Modules could be imported either seperately or all together.

```python
from MWDC import *

## or ##

from MWDC import transformation
from MWDC import evaluation
.
.
.

```

## Modules Documentation

### preprocessing

| Functions                   | Description                                                 |
| :-------------------------- | :---------------------------------------------------------- |
| `transformddaily(x)`        | Transformation function for Daily Data                      |
| `transformdmock(x)`         | Transformation function for Mock Data                       |
| `transformqm(x)`            | Variable for Quater Map                                     |
| `datatransformation(input)` | Description in the Note below\*                             |
| `datanormalization(input)`  | Input in this case will be the transformed pandas dataframe |
| `null_fill(input)`          | Function to input NaN values across variables               |
| `pca1(data, n)`             | data is data to be input , n is the number of components    |

\*Note: This function is used to transform the xarray dataset into a pandas dataframe where the dimension "time" would become the index of the DataFrame and,
pairs of both dimensions "latitude" and "longitude" will become the columns for each variable

### clustering

#### - DBscan

| Functions                        | Description                                                     |
| :------------------------------- | :-------------------------------------------------------------- |
| `dbscanreal(x, eps1=0.5, min=5)` | eps1 for epsilon , min for minimum samples, x is for data input |

#### - Kmeans

| Functions                                                                      | Description |
| :----------------------------------------------------------------------------- | :---------- |
| `Kmeans(n_cluster).fit(xarray_data, PCA=(boolian), pass_trans_data=(boolian))` | \*          |
| `Kmeans(n_cluster).evaluate(z, PCA=(boolian), pass_trans_data=(boolian))`      | \*\*        |

\* This function fits the K-means model to the data that is passed to it.  
 Parameters that this function will accept are as follows:  
 1. xarray_data = string of the name of the original xarray file  
 2. PCA (bool) = whether or not PCA has to be applied. Default value is True.  
 3. pass_trans_data (bool) = whether saved data has to be passed. If False, data will be transformed instantly. Default value is True.

\*\* This function evaluates and assigns data points to clusters.
Parameters that this function will accept are as follows:  
 1. z = string of the name of the original xarray file.  
 2. PCA (bool) = whether or not PCA has to be applied. Default value is True.  
 3. pass_trans_data (bool) = whether saved data has to be passed. If False, data will be transformed instantly. Default value is True.

#### - evaluation

| Functions             | Params                                                                           |
| :-------------------- | :------------------------------------------------------------------------------- |
| `RMSE()`              | input, formed_clusters, frame, normalize=False                                   |
| `Spat_Corr()`         | input, formed_clusters, trans_data, normalize=False                              |
| `silhouette_score1()` | X, labels, \*, metric="euclidean", sample_size=None, random_state=None, \*\*kwds |

#### - visualization

| Functions                 | Params                                                                     |
| :------------------------ | :------------------------------------------------------------------------- |
| `visualization()`         | data_file,cluster_filename,coast_file                                      |
| `make_Csv_cluster()`      | label,name                                                                 |

## Parameters that this function will accept are as follows:   
    * visualization()   
    1. data_file is the .nc file.   
    Example data_file = 'path/data.nc'  It is the raw unprocessed data.   
    
    2. cluster_filename is the csv file which contains clusterid and time_step.    
    Example cluster_filename = 'path/clusters.csv'  # This file contains what cluster belongs to what date.    
    
    3. coast_file =  This file contains the data of how a coastline should look like in the result.     
    Example 'path/coast.txt'. 
    
    * make_Csv_cluster().   
    1.# label contains the clusterids.   
    2. # name is the file name that will generated eg:('test.csv').   
    
    
