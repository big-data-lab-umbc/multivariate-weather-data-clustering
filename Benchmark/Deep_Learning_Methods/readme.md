# Deep Spatiotemporal Clustering: A Temporal Clustering Approach for Multi-dimensional Climate Data

Authors: Omar Faruque, Francis Ndikum Nji, Mostafa Cham, Rohan Mandar Salvi, Xue Zheng, and Jianwu Wang


## Abstract
Clustering high-dimensional spatiotemporal data using an unsupervised approach is a challenging problem for many data-driven applications. Existing state-of-the-art methods for unsupervised clustering use different similarity and distance functions but focus on either spatial or temporal features of the data. Concentrating on joint deep representation learning of spatial and temporal features, we propose Deep Spatiotemporal Clustering (DSC), a novel algorithm for the temporal clustering of high-dimensional spatiotemporal data using an unsupervised deep learning method. Inspired by the U-net architecture, DSC utilizes an autoencoder integrating CNN-RNN layers to learn latent representations of the spatiotemporal data. DSC also includes a unique layer for cluster assignment on latent representations that uses the Student's t-distribution. By optimizing the clustering loss and data reconstruction loss simultaneously, the algorithm gradually improves clustering assignments and the nonlinear mapping between low-dimensional latent feature space and high-dimensional original data space. A multivariate spatiotemporal climate dataset is used to evaluate the efficacy of the proposed method. Our extensive experiments show our approach outperforms both conventional and deep learning-based unsupervised clustering algorithms. Additionally, we compared the proposed model with its various variants (CNN encoder, CNN autoencoder, CNN-RNN encoder, CNN-RNN autoencoder, etc.) to get insight into using both the CNN and RNN layers in the autoencoder, and our proposed technique outperforms these variants in terms of clustering results.

## Dataset
For this study, we use the open-access atmospheric reanalysis data  from European Centre for Medium-Range Weather Forecasts (ECMWF) ERA-5 global reanalysis product. Seven atmospheric reanalysis variables from the ERA5 dataset were selected for this study. These variables are included in the dataset based on their impact on the air-sea-cloud interaction system and were measured in a latitude-longitude grid of (41x41). Temporally, the dataset covers one year period and one observation per day. The variables are mentioned in the following table.
| Variable                   | Short name |  Unit |  Range |
| :---:                      |   :---:    | :---: | :---: |
| Sea Surface Temperature    | sst        | k     | 285-300 |
| 2 meter Air Temperature    | t2m        | k     | 281 to 299 |
| Surface Pressure           | sp         | pa    | 98260 to 103788 |
| Surface Sensible Heat Flux | sshf       | J/m^2 | -674528 to 200024 |
| Surface Latent Heat Flux   | slhf       | J/m^2 | -1840906 to 90131|
| 10-meter U wind            | u10        | m/s   | -16 to 19 |
| 10-meter V wind            | v10        | m/s   | -15 to 16 |

## Code
This repository contains the core implementation of our proposed model and all the baseline models we have compared for performance evaluation. Also, the ablation study codes are available here. A brief summary of all notebooks is given below. 

### Deep Spatiotemporal Clustering (DSC)
The proposed Deep Spatiotemporal Clustering method implementation is available in this notebook: DSC_Clustering_Model.ipynb.




