######## PCA #########
import pandas as pd
from sklearn.decomposition import PCA

def pca1(data,n): # data is data to be input , n is the number of components 
  pca = PCA(n_components=n) 
  pca.fit(data)

  # Get pca scores
  pca_scores = pca.transform(data)

  # Convert pca_scores to a dataframe
  scores_df = pd.DataFrame(pca_scores)

  # Round to two decimals
  scores_df = scores_df.round(2)

  # Return scores
  return scores_df

######## End of PCA #########