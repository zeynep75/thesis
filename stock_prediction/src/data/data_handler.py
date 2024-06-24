# pylint: disable=import-error, W0718, C0301

'''
This module is responsible for handling the data. It will be responsible for loading 
the data and handling the data
'''

# Third-party libraries
from sklearn.model_selection import train_test_split
import pandas as ps

# Custom libraries
from stock_prediction.src.config_loader import load_config
from stock_prediction.src.features import create_features, remove_unused_features
from .preprocessing import prepare_data, normalize_data, preprocess_data

def load_data():
    '''
    This function is used to load the data
    '''
    
    # Load the configuration file
    config = load_config()
    
    # Check if the configuration file was loaded
    if config is None:
        return None
    
    # Index of the dataset to be used [0: 5 years, 1: 10 years]
    dataset_index = config['data']['dataset_index']

    # Defining the paths to the datasets
    five_dataset_path = config['data']['five_years_path']
    ten_dataset_path = config['data']['ten_years_path']

    # Defining the path to the dataset to be used
    active_dataset_path = five_dataset_path if dataset_index == 0 else ten_dataset_path
    
    try:
        # Load the dataset
        dataset = ps.read_csv(active_dataset_path, parse_dates=['Date'], index_col='Date')
    except Exception as e:
        print(f'Error: {e}')
        return None
    
    # Return the dataset
    return dataset

def handle_data(dataset : ps.DataFrame):
    '''
    This function is used to handle the dataset, preprocess it, and split into test, validation, training groups
    '''
    
    # Load the configuration file
    config = load_config()
    
    if config is None:
        return None
    
    df = create_features(dataset)
    df = remove_unused_features(df)
    df = preprocess_data(df, 'b_fill', drop=True)
    
    df_features, features_scaler = normalize_data(df, df.columns.drop('Close'))
    df_target, target_scaler = normalize_data(df, ['Close'])

    if config['model']['model_type'] == "LSTM":
        # Prepare the data 
        x,y = prepare_data(df_features.values, df_target.columns.get_loc('Close'), config['model']['training_settings']['time_steps'])
    else:
        x = df_features.drop('Close', axis=1).values
        y = df_target['Close'].values
        
    # Create the training and testing and validation datasets (80% train, 10% test, 10% validation)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, shuffle=False)

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "x_val": x_val,
        "y_val": y_val
    }, {
        "features_scaler": features_scaler,
        "target_scaler": target_scaler
    }
