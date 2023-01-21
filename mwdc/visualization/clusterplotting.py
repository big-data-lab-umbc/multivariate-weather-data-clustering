import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def clusterPlot2D(data, generatedCluser):
  tsne = TSNE(n_components=2, learning_rate='auto', perplexity=10)
  tsne_data = tsne.fit_transform(data)
  tsne_data
  tsne_df = pd.DataFrame(tsne_data, columns=['Feature-1','Feature-2'])
  tsne_df['cluster'] = pd.Categorical(generatedCluser)
  sns.scatterplot(x="Feature-1",y="Feature-2",hue="cluster",data=tsne_df)
  plt.legend(title='Clusters', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
