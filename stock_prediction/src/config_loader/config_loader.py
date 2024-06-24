# pylint: disable=W0703

'''
This file is used to load the configuration file
'''

# Default libraries
import os
import json
import pathlib

def load_config():
    '''
    This function is used to load the configuration file
    '''
    
    # Get the path of the configuration file
    config_path = os.path.join(pathlib.Path(__file__).parents[3].absolute(), 'config.json')
    
    try:
        # Load the configuration file
        with open(config_path, encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        print(f'Error: {e}')
        return None
    
    return config

def fetch_active_features(config = None):
    '''
    This function is used to fetch the active features from the configuration file
    '''
        
    # Check if the configuration file was loaded
    if config is None:
        config = load_config()
    
    # Check if the configuration file is still None
    if config is None:
        return None
    
    # Fetch the features
    features = config['data']['features']

    # Initialize the active features list
    active_features = []
    
    # Loop through the features and fetch the active features
    for feature in features:
        if feature['is_used']:
            active_features.append(feature['name'])
    
    return active_features

if __name__ == '__main__':
    load_config()
    print(fetch_active_features(load_config()))
