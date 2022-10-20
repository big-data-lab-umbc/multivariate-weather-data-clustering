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

#### - clustering

| Functions                        | Description                                                     |
| :------------------------------- | :-------------------------------------------------------------- |
| `dbscanreal(x, eps1=0.5, min=5)` | eps1 for epsilon , min for minimum samples, x is for data input |

#### - dim_reduction

| Functions       | Description                                              |
| :-------------- | :------------------------------------------------------- |
| `pca1(data, n)` | data is data to be input , n is the number of components |

#### - evaluation

| Functions             | Params                                                                           |
| :-------------------- | :------------------------------------------------------------------------------- |
| `RMSE()`              | input, formed_clusters, frame, normalize=False                                   |
| `Spat_Corr()`         | input, formed_clusters, trans_data, normalize=False                              |
| `silhouette_score1()` | X, labels, \*, metric="euclidean", sample_size=None, random_state=None, \*\*kwds |

#### - normalization

| Functions                  | Description                                                 |
| :------------------------- | :---------------------------------------------------------- |
| `datanormalization(input)` | Input in this case will be the transformed pandas dataframe |

#### - physics_based

| Functions          | Description                                   |
| :----------------- | :-------------------------------------------- |
| `null_fill(input)` | Function to input NaN values across variables |

#### - transformation

| Functions                   | Description                            |
| :-------------------------- | :------------------------------------- |
| `transformddaily(x)`        | Transformation function for Daily Data |
| `transformdmock(x)`         | Transformation function for Mock Data  |
| `transformqm(x)`            | Variable for Quater Map                |
| `datatransformation(input)` | Description in the Note below\*        |

\*Note: This function is used to transform the xarray dataset into a pandas dataframe where the dimension "time" would become the index of the DataFrame and,
pairs of both dimensions "latitude" and "longitude" will become the columns for each variable

#### - visualization

| Functions                 | Params                                                                     |
| :------------------------ | :------------------------------------------------------------------------- |
| `plot_map()`              | var, var_range, lon0, lat0, fig ,panel ,cmap0, colorbar, title, ifcontourf |
| `read_combined_cluster()` | csvlink, varid                                                             |
| `plotcoastline()`         | file_link, color='k'                                                       |
