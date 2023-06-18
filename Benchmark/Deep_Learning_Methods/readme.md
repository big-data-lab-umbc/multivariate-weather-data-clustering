# Deep Spatiotemporal Clustering: A Temporal Clustering Approach for Multi-dimensional Climate Data

Authors: Omar Faruque, Francis Ndikum Nji, Mostafa Cham, Rohan Mandar Salvi, Xue Zheng, and Jianwu Wang

## Citation
If you use this code for your research, please cite our paper.

## Abstract
Clustering high-dimensional spatiotemporal data using an unsupervised approach is a challenging problem for many data-driven applications. Existing state-of-the-art methods for unsupervised clustering use different similarity and distance functions but focus on either spatial or temporal features of the data. Concentrating on joint deep representation learning of spatial and temporal features, we propose Deep Spatiotemporal Clustering (DSC), a novel algorithm for the temporal clustering of high-dimensional spatiotemporal data using an unsupervised deep learning method. Inspired by the U-net architecture, DSC utilizes an autoencoder integrating CNN-RNN layers to learn latent representations of the spatiotemporal data. DSC also includes a unique layer for cluster assignment on latent representations that uses the Student's t-distribution. By optimizing the clustering loss and data reconstruction loss simultaneously, the algorithm gradually improves clustering assignments and the nonlinear mapping between low-dimensional latent feature space and high-dimensional original data space. A multivariate spatiotemporal climate dataset is used to evaluate the efficacy of the proposed method. Our extensive experiments show our approach outperforms both conventional and deep learning-based unsupervised clustering algorithms. Additionally, we compared the proposed model with its various variants (CNN encoder, CNN autoencoder, CNN-RNN encoder, CNN-RNN autoencoder, etc.) to get insight into using both the CNN and RNN layers in the autoencoder, and our proposed technique outperforms these variants in terms of clustering results.

## Dataset
For this study, we use the open-access atmospheric reanalysis data  from European Centre for Medium-Range Weather Forecasts (ECMWF) ERA-5 global reanalysis product [2]. Seven atmospheric reanalysis variables from the ERA5 dataset were selected for this study. These variables are included in the dataset based on their impact on the air-sea-cloud interaction system and were measured in a latitude-longitude grid of (41x41). Temporally, the dataset covers one year period and one observation per day. The variables are mentioned in the following table.
| Variable                   | Short name |  Unit   |  Range |
| :---:                      |   :---:    |  :---:  | :---: |
| Sea Surface Temperature    | sst        | $k$     | 285-300 |
| 2 meter Air Temperature    | t2m        | $k$     | 281 to 299 |
| Surface Pressure           | sp         | $pa$    | 98260 to 103788 |
| Surface Sensible Heat Flux | sshf       | $J/m^2$ | -674528 to 200024 |
| Surface Latent Heat Flux   | slhf       | $J/m^2$ | -1840906 to 90131|
| 10-meter U wind            | u10        | $m/s$   | -16 to 19 |
| 10-meter V wind            | v10        | $m/s$   | -15 to 16 |

## Code
This repository contains the code implementation of our proposed model and all the baseline models we have compared for performance evaluation. Also, the ablation study codes are available here. A brief summary of all notebooks is given below. 

- DSC_Clustering_Model.ipynb: The proposed Deep Spatiotemporal Clustering method implementation is available in this notebook.
- DEC-Clustering-Model.ipynb: The Deep Embedded Clustering (DEC)[4] baseline model is applied to our dataset in this notebook.  
- DTC-Clustering-Model.ipynb: The Deep Temporal Clustering (DTC)[5] baseline model is applied to our dataset in this notebook.
- K-Means_And_Hierarchical_Clustering_Algorithms.ipynb: In this notebook, we applied the k-means and hierarchical clustering algorithm on the dataset as the state-of-the-art method.   
- CNN_LSTM_Encoder.ipynb: In this notebook, we only applied the encoder part of the proposed model on the dataset. It is part of the ablation study to find the effectiveness of the proposed autoencoder model.
- CNN_Autoencoder_Model.ipynb: Here we have developed an autoencoder model using CNN deep learning layers and applied it to our dataset for clustering. 
- CNN_Encoder_Model.ipynb: In this notebook, the encoder part of the CNN autoencoder model is applied for clustering to compare  the clustering results of the CNN Autoencoder and Encoder model.


## References
1. Takahashi, N., Hayasaka, T.: Air–Sea Interactions among Oceanic Low-Level Cloud, Sea Surface Temperature, and Atmospheric Circulation on an Intraseasonal Time Scale in the Summertime North Pacific Based on Satellite Data Analysis. In: Journal of Climate, 33(21), 9195-9212. Retrieved Feb 6, 2023, from https://journals.ametsoc.org/view/journals/clim/33/21/jcliD190670.xml.
2. European Centre for Medium-Range Weather Forecasts. ERA-5 global reanalysis product. https://cds.climate.copernicus.eu/cdsapp#!/home, 2021. Last Accessed: 2021-9-5.
3. Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18 (pp. 234-241). Springer International Publishing.
4. Xie, J., Girshick, R., Farhadi, A.: Unsupervised deep embedding for clustering analysis. In: International conference on machine learning. PMLR, 2016.
5. Madiraju, N. S., Sadat, S. M., Fisher, D., Karimabadi, H.: Deep Temporal Clustering: Fully Unsupervised Learning of Time-Domain Features. In: ArXiv (2018). https://doi.org/10.48550/arXiv.1802.01059.
6. Ghosh, R., Jia, X., Yin, L., Lin, C., Jin, Z., Kumar, V.: Clustering augmented self-supervised learning: an application to land cover mapping. In: 30th International Conference on Advances in Geographic Information Systems (SIGSPATIAL’22). Association for Computing Machinery, New York, NY, USA, Article 3, 1–10. https://doi.org/10.1145/3557915.3560937.
7. Ma, Q., Zheng, J., Li, S., Cottrell, G. W.: Learning representations for time series clustering. In: 33rd International Conference on Neural Information Processing System (2019). Curran Associates Inc., Red Hook, NY, USA, Article 339, 3781–3791.
8. Hadifar, A., Sterckx, L., Demeester, T., Develder, C.: A self-training approach for short text clustering. In: 4th Workshop on Representation Learning for NLP (RepL4NLP-2019) (pp. 194-199), Florence, Italy (2019, August).
 




