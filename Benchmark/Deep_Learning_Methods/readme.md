# Deep Spatiotemporal Clustering: A Temporal Clustering Approach for Multi-dimensional Climate Data

## Code
This repository contains the code implementation of our proposed model and all the baseline models we have compared for performance evaluation. Also, the ablation study codes are available here. A brief summary of all notebooks is given below. 

- DSC_Clustering_Model.ipynb: The proposed Deep Spatiotemporal Clustering method implementation is available in this notebook.
- DEC-Clustering-Model.ipynb: The Deep Embedded Clustering (DEC)[4] baseline model is applied to our dataset in this notebook.  
- DTC-Clustering-Model.ipynb: The Deep Temporal Clustering (DTC)[5] baseline model is applied to our dataset in this notebook.  
- CNN_LSTM_Encoder.ipynb: In this notebook, we only applied the encoder part of the proposed model on the dataset. It is part of the ablation study to find the effectiveness of the proposed autoencoder model.
- CNN_Autoencoder_Model.ipynb: Here we have developed an autoencoder model using CNN deep learning layers and applied it to our dataset for clustering. 
- CNN_Encoder_Model.ipynb: In this notebook, the encoder part of the CNN autoencoder model is applied for clustering to compare  the clustering results of the CNN Autoencoder and Encoder model.



 





