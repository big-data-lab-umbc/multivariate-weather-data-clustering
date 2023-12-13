# multivariate-weather-data-clustering for HESC branch

## Download

There are three ways to Download and Manage the MWDC package:

1 - Use [GitHub Desktop](https://desktop.github.com/) (Recomended)

2 - Use command line:

```bash
 git clone https://github.com/big-data-lab-umbc/multivariate-weather-data-clustering.git
```

\*Because the repository is private the command line method is not Recomended.

3 - Download the `.zip` file and use it.

4 - On Google Colab use the command below.

```bash
!git clone https://{clasic_access_token}@github.com/big-data-lab-umbc/multivariate-weather-data-clustering.git
```

\*\* This is how to generat [clasic_access_token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token#creating-a-personal-access-token-classic).

## Installation

#### 1. On PC

To install the package you need to create an environment using [pip](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) or [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

##### Conda environment setup
```bash
conda create -n mwdc pandas numpy xarray netCDF4 matplotlib scikit-learn scipy dask
conda activate mwdc
```

After that just clone this repository and install the ` setup.py` file inside it.

```bash
 cd multivariate-weather-data-clustering
 python setup.py install
```

Note: If you are using macOS, you should use ` python3 setup.py install` instead.

#### 2. On Google Colab

After cloning the repository just run the command below to install it.

```bash
 %cd multivariate-weather-data-clustering
 !python setup.py install
```

## Usage

To use the functions you just need to import them from MWDC. Modules could be imported either seperately or all together.

```python
from mwdc import *

## or ##

from mwdc.preprocessing import preprocessing
from mwdc.evaluation import st_evaluation
from mwdc.visualization import visualization

```

Example:

```python
trans_data = preprocessing.datatransformation(data)
```

## Modules Documentation

### preprocessing

| Functions              | Description                                                                      |
| :--------------------- | :------------------------------------------------------------------------------- |
| `transformddaily()`    | Transformation function for Daily Data                                           |
| `transformdmock()`     | Transformation function for Mock Data                                            |
| `transformqm()`        | Variable for Quater Map                                                          |
| `datatransformation()` | Description in the Note below\*                                                  |
| `datanormalization()`  | Input in this case will be the transformed pandas dataframe                      |
| `null_fill()`          | Function to input NaN values across variables                                    |
| `pca1()`               | data is data to be input , n is the number of components                         |
| `pcacomponents()`      | Showing the proper number of components for pca by computing cumulative variance |
| `data_preprocessing()` | Transforms the xArray input data into a 2D NumPy Array.                          |

\*Note: This function is used to transform the xarray dataset into a pandas dataframe where the dimension "time" would become the index of the DataFrame and,
pairs of both dimensions "latitude" and "longitude" will become the columns for each variable

### clustering

#### - DBscan

| Functions                        | Description                                                     |
| :------------------------------- | :-------------------------------------------------------------- |
| `dbscanreal(x, eps1=0.5, min=5)` | eps1 for epsilon , min for minimum samples, x is for data input |

#### - Agglomerative Clustering

| Functions                        | Description                                                     |
| :------------------------------- | :-------------------------------------------------------------- |
| `st_agglomerative(data, n, K, p=7, affinity, linkage)| n=PCA components, K=number of clusters, p=truncate_mode.

#### - Kmeans

| Functions                                                                      | Description |
| :----------------------------------------------------------------------------- | :---------- |
| `Kmeans(n_cluster).fit(xarray_data, PCA=(boolian), pass_trans_data=(boolian))` | \*          |
| `Kmeans(n_cluster).evaluate(z, PCA=(boolian), pass_trans_data=(boolian))`      | \*\*        |

\* This function fits the K-means model to the data that is passed to it.  
 Parameters that this function will accept are as follows:

1.  xarray_data = string of the name of the original xarray file
2.  PCA (bool) = whether or not PCA has to be applied. Default value is True.
3.  pass_trans_data (bool) = whether saved data has to be passed. If False, data will be transformed instantly. Default value is True.

\*\* This function evaluates and assigns data points to clusters.
Parameters that this function will accept are as follows:

1.  z = string of the name of the original xarray file.
2.  PCA (bool) = whether or not PCA has to be applied. Default value is True.
3.  pass_trans_data (bool) = whether saved data has to be passed. If False, data will be transformed instantly. Default value is True.

#### - evaluation

| Functions                    | Params                                                                                                |
| :--------------------------- | :---------------------------------------------------------------------------------------------------- |
| `st_rmse()`                  | input,formed_clusters                                                                                 |
| `st_corr()`                  | input,formed_clusters                                                                                 |
| `st_calinski()`              | input,formed_clusters                                                                                 |
| `davies_bouldin()`           | input, formed_clusters                                                                                |
| `compute_silhouette_score()` | X, labels,transformation=False, \*, metric="euclidean", sample_size=None, random_state=None, \*\*kwds |

#### - visualization

| Functions            | Params                                |
| :------------------- | :------------------------------------ |
| `visualization()`    | data_file,cluster_filename,coast_file |
| `make_Csv_cluster()` | label,name                            |

\* Parameters that `visualization()` will accept are as follows:

1.  data_file is the .nc file.  
    \- Example data_file = 'path/data.nc' It is the raw unprocessed data.
2.  cluster_filename is the csv file which contains clusterid and time_step.  
    \- Example cluster_filename = 'path/clusters.csv' # This file contains what cluster belongs to what date.
3.  coast_file = This file contains the data of how a coastline should look like in the result.  
    \- Example 'path/coast.txt'.

####

\* Parameters that `make_Csv_cluster()` will accept are as follows:

1.  label contains the clusterids.
2.  Name is the file name that will generated eg:('test.csv').
