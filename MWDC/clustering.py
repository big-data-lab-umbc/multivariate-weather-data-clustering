################## DBscan ##################
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

def dbscanreal(x,eps1=0.5,min=5): # eps1 for epsilon , min for minimum samples, x is for data input
      db = DBSCAN(eps=eps1,min_samples=min,metric='cosine').fit(x)  # cosine distance calculation
      core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
      core_samples_mask[db.core_sample_indices_] = True
      labels = db.labels_

      # Number of clusters in labels, ignoring noise if present.
      n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
      n_noise_ = list(labels).count(-1)
      frame = pd.DataFrame(x)
      frame['Cluster'] = labels
      frame['Cluster'].value_counts()
      print("Estimated number of clusters: %d" % n_clusters_)
      print(frame['Cluster'].value_counts())
      print("Estimated number of noise points: %d" % n_noise_)
      #print("Silhouette Coefficient: %0.3f" % silhouette_score(x, labels))
      return frame,labels