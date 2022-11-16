from sklearn_extra.cluster import KMedoids
import pandas as pd

def kmedoids(X,data): # X is the number of n_clusters, 'data' is for data input
    
      kmedoids = KMedoids(n_clusters=X, metric='cosine',init='k-medoids++', ).fit(data)  # cosine distance calculation
      labels = kmedoids.labels_
      frame = pd.DataFrame(data)
      frame['Cluster'] = labels
      frame['Cluster'].value_counts()
      print("Estimated number of clusters: %d" % X)
      print(frame['Cluster'].value_counts())
      #print("Silhouette Coefficient: %0.3f" % silhouette_score(x, labels))
      return frame,labels