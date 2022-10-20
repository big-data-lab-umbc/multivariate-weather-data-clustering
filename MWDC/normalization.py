
##### Data Normilization #####
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def datanormalization(input):
    ''' This function is used to normalize the data that is passed to it. Input in this case will be the transformed pandas dataframe. '''
    x = input.values # returns a numpy array
    min_max_scaler = MinMaxScaler() # calling the function
    x_scaled = min_max_scaler.fit_transform(x) # x_scaled will hold the values of the normalized data
    
    # trans_data will hold the same columns and index of the dataframe that is passed to it. And the values will be the ones saved in x_scaled
    trans_data = pd.DataFrame(x_scaled, columns=input.columns, index=input.index)
        
    return trans_data