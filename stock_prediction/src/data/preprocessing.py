# pylint: disable=import-error, C0301

'''
This module is used to preprocess the data for the project. This includes cleaning and transforming the data.
'''

# Third-party libraries
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Custom Imports
from stock_prediction.src.config_loader import fetch_active_features

# Normalize the dataset
def normalize_data(df, cols):
    '''
    Normalizes the data in the dataframe
    '''
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])
    
    return df, scaler



def prepare_data(df, target_col, time_steps=60):
    '''
    Prepares the data for training the LSTM model
    '''
    
    x, y = [], []
    for i in range(len(df) - time_steps):
        x.append(df[i:i + time_steps])
        y.append(df[i + time_steps, target_col])
    
    print(f'x: {len(x)} y: {len(y)}')
    print(f'x: {x[0]} y: {y[0]}')
    
    return np.array(x), np.array(y)

def preprocess_data(df, method='f_fill', drop = False, constant_fill=0.0):
    '''
    Preprocesses the data for training the LSTM model
    '''
    
    # Handle missing values
    if method == 'f_fill':
        df.fillna(method='ffill', inplace=True)    
    elif method == 'b_fill':
        df.fillna(method='bfill', inplace=True)
    elif method == 'constant':
        df.fillna(constant_fill, inplace=True)

    # Drop missing values
    if drop:
        df.dropna(inplace=True)

    # Check for missing values
    return df 