import pandas as pd
import numpy as np
import xarray as xr
import netCDF4 as nc

# Fit and Evaluate functions with PCA as a parameter (Luke, 2022)

def euclidean(point, data):
    """
    Euclidean distance between point & data.
    Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
    """
    return np.sqrt(np.sum((point - data)**2, axis=1))
    

class KMeans:
    def __init__(self, n_clusters, max_iter=500):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, xarray_data, PCA = True, pass_trans_data = True):

      ''' This function fits the K-means model to the data that is passed to it.
          Parameters that this function will accept are as follows:
          1. xarray_data = string of the name of the original xarray file
          2. PCA (bool) = whether or not PCA has to be applied. Default value is True
          3. pass_trans_data (bool) = whether saved data has to be passed. If False, data will be transformed instantly. Default value is True.'''

      # The following block will run if user wants PCA to be applied to the data
      if PCA == True:
        # The following code will be executed if the user wants the saved transformed dataframe to be considered.
        if pass_trans_data == True:
          k = list(locals().values()) # k will be the list which holds the values of the parameters that are being passed to the function
          path = str(k[1]) # path will hold the first parameter's value
          fullpath = os.path.join("/content/drive/MyDrive/Courses/IS-700-Independent_Study/Transformed_data/" + path + ".csv")

          X_train = pd.read_csv(fullpath, index_col=[0]) # saved transformed dataframe will be read
          X_train = datanormalization(X_train) # the data will be normalized
          X_train = np.array(PCA_transform(X_train)) # X_train in the form of a numpy array will hold the data to which PCA has been applied 

        

          ''' Initialize the centroids, using the "k-means++" method, where a random datapoint is selected as the first point,
              then the rest are initialized with probabilities proportional to their distances to the first point.'''
          
          # Pick a random point from train data for first centroid
          self.centroids = [random.choice(X_train)]

          for _ in range(self.n_clusters-1):
              # Calculate distances from points to the centroids
              dists = np.sum([euclidean(centroid, X_train) for centroid in self.centroids], axis=0)
              # Normalize the distances
              dists /= np.sum(dists)
              # Choose remaining points based on their distances
              new_centroid_idx, = np.random.choice(range(len(X_train)), size=1, p=dists)
              self.centroids += [X_train[new_centroid_idx]]

          iteration = 0 # setting iteration to 0
          prev_centroids = None # setting prev_centroids to None

          while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
              # Sort each datapoint, and assign it to the nearest centroid
              sorted_points = [[] for _ in range(self.n_clusters)]
              for x in X_train:
                  dists = euclidean(x, self.centroids) # Calling the function defined above
                  centroid_idx = np.argmin(dists) # Assigning points to the nearest centroid
                  sorted_points[centroid_idx].append(x)
              # Push current centroids to previous, reassign centroids as mean of the points belonging to them
              prev_centroids = self.centroids
              self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
              for i, centroid in enumerate(self.centroids):
                  if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                      self.centroids[i] = prev_centroids[i]
              iteration += 1

        # If the user does not want the saved transformed data to be used, the below block of code will be executed    
        else:
          X_train = datatransformation(xarray_data) # Transforming the data
          X_train = datanormalization(X_train) # Normalizing the data
          X_train = np.array(PCA_transform(X_train)) # Applying PCA to the data and storing it in the form of a numpy array

          ''' Initialize the centroids, using the "k-means++" method, where a random datapoint is selected as the first point,
              then the rest are initialized with probabilities proportional to their distances to the first point.'''
          
          # Pick a random point from train data for first centroid
          self.centroids = [random.choice(X_train)]
          for _ in range(self.n_clusters-1):
              # Calculate distances from points to the centroids
              dists = np.sum([euclidean(centroid, X_train) for centroid in self.centroids], axis=0)
              # Normalize the distances
              dists /= np.sum(dists)
              # Choose remaining points based on their distances
              new_centroid_idx, = np.random.choice(range(len(X_train)), size=1, p=dists)
              self.centroids += [X_train[new_centroid_idx]]

          iteration = 0 # setting iteration to 0
          prev_centroids = None # setting prev_centroids to None

          while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
              # Sort each datapoint, and assign it to the nearest centroid
              sorted_points = [[] for _ in range(self.n_clusters)]
              for x in X_train:
                  dists = euclidean(x, self.centroids) # Calling the function defined above
                  centroid_idx = np.argmin(dists) # Assigning points to the nearest centroid
                  sorted_points[centroid_idx].append(x)
              # Push current centroids to previous, reassign centroids as mean of the points belonging to them
              prev_centroids = self.centroids
              self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
              for i, centroid in enumerate(self.centroids):
                  if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                      self.centroids[i] = prev_centroids[i]
              iteration += 1      

      # If the user does not want PCA to be applied to the data, the below block of code will be executed
      else:
        # The following code will be executed if the user wants the saved transformed dataframe to be considered.
        if pass_trans_data == True:
          k = list(locals().values()) # k will be the list which holds the values of the parameters that are being passed to the function
          path = str(k[1]) # path will hold the first parameter's value
          fullpath = os.path.join("/content/drive/MyDrive/Courses/IS-700-Independent_Study/Transformed_data/" + path + ".csv")

          X_train = pd.read_csv(fullpath, index_col=[0]) # saved transformed dataframe will be read
          X_train = np.array(datanormalization(X_train)) # the data will be normalized and stored in the form of a numpy array

          ''' Initialize the centroids, using the "k-means++" method, where a random datapoint is selected as the first point,
          # then the rest are initialized with probabilities proportional to their distances to the first point. '''
          
          # Pick a random point from train data for first centroid
          self.centroids = [random.choice(X_train)]
          for _ in range(self.n_clusters-1):
              # Calculate distances from points to the centroids
              dists = np.sum([euclidean(centroid, X_train) for centroid in self.centroids], axis=0)
              # Normalize the distances
              dists /= np.sum(dists)
              # Choose remaining points based on their distances
              new_centroid_idx, = np.random.choice(range(len(X_train)), size=1, p=dists)
              self.centroids += [X_train[new_centroid_idx]]

          iteration = 0 # setting iteration to 0
          prev_centroids = None # setting prev_centroids to None

          while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
              # Sort each datapoint, and assign it to the nearest centroid
              sorted_points = [[] for _ in range(self.n_clusters)]
              for x in X_train:
                  dists = euclidean(x, self.centroids) # Calling the function defined above
                  centroid_idx = np.argmin(dists) # Assigning points to the nearest centroid
                  sorted_points[centroid_idx].append(x)
              # Push current centroids to previous, reassign centroids as mean of the points belonging to them
              prev_centroids = self.centroids
              self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
              for i, centroid in enumerate(self.centroids):
                  if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                      self.centroids[i] = prev_centroids[i]
              iteration += 1

        # If the user does not want the saved transformed data to be used, the below block of code will be executed        
        else:
          X_train = datatransformation(xarray_data) # Transforming the data
          X_train = np.array(datanormalization(X_train)) # Normalizing the data and storing it in the form of a numpy array

          ''' Initialize the centroids, using the "k-means++" method, where a random datapoint is selected as the first point,
           then the rest are initialized w/ probabilities proportional to their distances to the first point.'''

          # Pick a random point from train data for first centroid
          self.centroids = [random.choice(X_train)]
          for _ in range(self.n_clusters-1):
              # Calculate distances from points to the centroids
              dists = np.sum([euclidean(centroid, X_train) for centroid in self.centroids], axis=0)
              # Normalize the distances
              dists /= np.sum(dists)
              # Choose remaining points based on their distances
              new_centroid_idx, = np.random.choice(range(len(X_train)), size=1, p=dists)
              self.centroids += [X_train[new_centroid_idx]]

          iteration = 0 # setting iteration to 0
          prev_centroids = None # setting prev_centroids to None

          while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
              # Sort each datapoint, assign it to the nearest centroid
              sorted_points = [[] for _ in range(self.n_clusters)]
              for x in X_train:
                  dists = euclidean(x, self.centroids)
                  centroid_idx = np.argmin(dists)
                  sorted_points[centroid_idx].append(x)
              # Push current centroids to previous, reassign centroids as mean of the points belonging to them
              prev_centroids = self.centroids
              self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
              for i, centroid in enumerate(self.centroids):
                  if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                      self.centroids[i] = prev_centroids[i]
              iteration += 1


    def evaluate(self, z, PCA = True, pass_trans_data = True):
      ''' This function evaluates and assigns data points to clusters
          Parameters that this function will accept are as follows:
          1. z = string of the name of the original xarray file
          2. PCA (bool) = whether or not PCA has to be applied. Default value is True.
          3. pass_trans_data (bool) = whether saved data has to be passed. If False, data will be transformed instantly. Default value is True.'''
      
      # The following block of code will be executed if the user wants PCA to be applied to the data.
      if PCA == True:
        # The following code will be executed if the user wants the function to consider already saved transformed data.
        if pass_trans_data == True:
          k = list(locals().values()) # k will be the list that holds the values of all the parameters that are being passed to the function
          path = str(k[1]) # path will capture the value of the first parameter
          fullpath = os.path.join("/content/drive/MyDrive/Courses/IS-700-Independent_Study/Transformed_data/" + path + ".csv")

          X = pd.read_csv(fullpath,index_col=[0]) # Already saved transformed dataframe will be read
          Y = datanormalization(X) # Data will be normalized here
          Y = np.array(PCA_transform(Y)) # PCA will be applied to the data and stored as a numpy array.

          # Declaring empty lists that will later hold the values of class_center and classification (1d array output)
          centroid = []
          centroid_idx = []

          i=0
          for x in Y:
              dists = euclidean(x, self.centroids) # Calling the function defined earlier
              centroid_id = np.argmin(dists) # Assigning the points to their nearest centroid
              centroid.append(self.centroids[centroid_id]) # Appending the class_centers
              centroid_idx.append(centroid_id) # Appending the cluster labels

          # The following steps will save the transformed data so that it can be used for visualization at a later stage.
          # Converting the normalized data array into a pandas dataframe
          transformed_data = pd.DataFrame(Y, index=X.index)

          # Adding class centers and cluster numbers as columns to the dataframe
          transformed_data['clusterid'] = centroid_idx 

          # Rearranging the columns in the dataframe
          transformed_data = transformed_data[['clusterid']] #+ [c for c in transformed_data if c not in ['clusterid']]]
          transformed_data1 = transformed_data # Storing the data along with the index
          transformed_data = transformed_data.reset_index() # Resetting the index of the dataframe

          return centroid, centroid_idx, transformed_data

        # If the user does not want saved data to be considered by the function, the below block will be executed
        else:
          X = datatransformation(z) # Transforming the original data into a pandas dataframe
          Y = datanormalization(X) # Normalizing the transformed data
          Y = np.array(PCA_transform(Y)) # Applying PCA to the data and storing it as a numpy array

          # Declaring empty lists that will later hold the values of class_center and classification (1d array output)
          centroid = []
          centroid_idx = []
          i=0
          for x in Y:
              dists = euclidean(x, self.centroids) # Calling the function defined earlier
              centroid_id = np.argmin(dists) # Assigning the points to their nearest centroid
              centroid.append(self.centroids[centroid_id]) # Appending the class_centers
              centroid_idx.append(centroid_id) # Appending the cluster labels

          # The following steps will save the transformed data so that it can be used for visualization at a later stage.
          # Converting the normalized data array into a pandas dataframe
          transformed_data = pd.DataFrame(Y, index=X.index)

          # Adding class centers and cluster numbers as columns to the dataframe
          transformed_data['clusterid'] = centroid_idx

          # Rearranging the columns in the dataframe
          transformed_data = transformed_data[['clusterid']]# + [c for c in transformed_data if c not in ['clusterid']]]
          transformed_data1 = transformed_data
          transformed_data = transformed_data.reset_index()

          return centroid, centroid_idx, transformed_data

      # The following block of code will be executed if the user does not want PCA to be applied to the data.
      else:
        # If the user wants already saved transformed data to be considered, the below code will be executed.
        if pass_trans_data == True:
          k = list(locals().values()) # k will be the list that holds the values of all the parameters that are being passed to the function
          path = str(k[1]) # path will hold the value of the first parameter
          fullpath = os.path.join("/content/drive/MyDrive/Courses/IS-700-Independent_Study/Transformed_data/" + path + ".csv")

          X = pd.read_csv(fullpath,index_col=[0]) # Already saved transformed data will be read
          Y = np.array(datanormalization(X)) # The data will be normalized and saved as a numpy array.

          # Declaring empty lists that will later hold the values of class_center and classification (1d array output)
          centroid = []
          centroid_idx = []

          i=0
          for x in Y:
              dists = euclidean(x, self.centroids) # Calling the function defined earlier
              centroid_id = np.argmin(dists) # Assigning the points to their nearest centroid
              centroid.append(self.centroids[centroid_id]) # Appending the class_centers
              centroid_idx.append(centroid_id) # Appending the cluster labels

          # The following steps will save the transformed data so that it can be used for visualization at a later stage.
          # Converting the normalized data array into a pandas dataframe
          transformed_data = pd.DataFrame(Y, index=X.index)

          # Adding class centers and cluster numbers as columns to the dataframe
          # transformed_data['Class_Center'] = centroid
          transformed_data['clusterid'] = centroid_idx   

          # Rearranging the columns in the dataframe
          transformed_data = transformed_data[['clusterid']]# + [c for c in transformed_data if c not in ['clusterid']]]
          transformed_data1 = transformed_data
          transformed_data = transformed_data.reset_index()

          return centroid, centroid_idx, transformed_data

        # If the user does not want saved data to be considered by the function, the below block will be executed
        else:
          X = datatransformation(z) # Transforming the original data into a pandas dataframe
          Y = np.array(datanormalization(X)) # Normalizing the data and saving it as a numpy array

          # Declaring empty lists that will later hold the values of class_center and classification (1d array output)
          centroid = []
          centroid_idx = []
          i=0
          for x in Y:
              dists = euclidean(x, self.centroids) # Calling the function defined earlier
              centroid_id = np.argmin(dists) # Assigning the points to their nearest centroid
              centroid.append(self.centroids[centroid_id]) # Appending the class_centers
              centroid_idx.append(centroid_id) # Appending the cluster labels

          # The following steps will save the transformed data so that it can be used for visualization at a later stage.
          # Converting the normalized data array into a pandas dataframe
          transformed_data = pd.DataFrame(Y, index=X.index)

          # Adding cluster numbers as a column to the dataframe
          transformed_data['clusterid'] = centroid_idx

          # Rearranging the columns in the dataframe
          transformed_data = transformed_data[['clusterid']] #+ [c for c in transformed_data if c not in ['clusterid']]]
          transformed_data1 = transformed_data
          transformed_data = transformed_data.reset_index()              

          return centroid, centroid_idx, transformed_data