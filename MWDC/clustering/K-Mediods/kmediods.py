from sklearn_extra.cluster import KMedoids
import pandas as pd
import matplotlib.pyplot as plt

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

def optimalk(data):
  cost =[]
  for i in range(1, 11):
      KM =KMedoids(n_clusters = i, metric='cosine',init='k-medoids++', max_iter = 500)
      KM.fit(data)
      
      # calculates squared error
      # for the clustered points
      cost.append(KM.inertia_)    
  
  # plot the cost against K values
  plt.plot(range(1, 11), cost, color ='g', linewidth ='3')
  plt.xlabel("Value of K")
  plt.ylabel("Squared Error (Cost)")
   # clear the plot
  return plt.show()