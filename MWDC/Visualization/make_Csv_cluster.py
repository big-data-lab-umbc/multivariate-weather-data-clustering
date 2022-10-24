import  pandas as pd
import numpy as np
def make_Csv_cluster(label,name):  # label contains the clusterids and name is the file name that will generated eg:('test.csv')
    df=pd.DataFrame()
    df1=pd.DataFrame()
    df['time_step'] = np.arange(len(label))
    df['clusterid'] = pd.DataFrame(label)
    df.to_csv(name,index=True)
    return