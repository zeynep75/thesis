# pylint: disable=line-too-long, import-error, C0301, E0611

'''
This module handles the tuning of the models. It is used to tune the hyperparameters of the models.
'''

# Default Libraries
import os
from typing import Union

# Third-party libraries
from kerastuner.tuners import RandomSearch
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
import joblib
import numpy as np

# Custom Libraries
from stock_prediction.src.config_loader import load_config
from .model_handler import LSTMModelHandler

def convert_svr_params(config: dict) -> dict:
    '''
    Convert the SVR hyperparameters to the correct format.
    '''
    
    # Convert the hyperparameters (looks ugly but it works!).
    # What happens is pretty much converts the hyperparameters to the correct format (list of values).
    # We use numpy to generate the values and then convert them to a list.
    converted_params = {
        'C': np.linspace(config["C"]["min"], config["C"]["max"], num=int((config["C"]["max"] - config["C"]["min"]) / config["C"]["step"])).tolist(),
        'gamma': np.linspace(config["gamma"]["min"], config["gamma"]["max"], num=int((config["gamma"]["max"] - config["gamma"]["min"]) / config["gamma"]["step"])).tolist(),
        'epsilon': np.linspace(config["epsilon"]["min"], config["epsilon"]["max"], num=int((config["epsilon"]["max"] - config["epsilon"]["min"]) / config["epsilon"]["step"])).tolist(),
        'degree': [int(x) for x in np.linspace(config["degree"]["min"], config["degree"]["max"], num=int((config["degree"]["max"] - config["degree"]["min"]) / config["degree"]["step"])).tolist()],
        'coef0': np.linspace(config["coef0"]["min"], config["coef0"]["max"], num=int((config["coef0"]["max"] - config["coef0"]["min"]) / config["coef0"]["step"])).tolist(),
        'tol': np.linspace(config["tol"]["min"], config["tol"]["max"], num=int((config["tol"]["max"] - config["tol"]["min"]) / config["tol"]["step"])).tolist(),
        'kernel': [config["kernel"]],
        'shrinking': [config["shrinking"]],
        'cache_size': [config["cache_size"]],
        'max_iter': [config["max_iter"]]
    }
    
    return converted_params

def tune_hyperparameters(model: Union[LSTMModelHandler, SVR], data_split: dict):
    '''
    This function is used to tune the hyperparameters of the model.

    Parameters:
        model: HyperModel or SVR
            The model to tune.
        data: dict
            The data to use for tuning the model.
            format: {'x_train': np.ndarray, 'y_train': np.ndarray, 'x_val': np.ndarray, 'y_val': np.ndarray}
    '''
    
    # Get the training, validation, and test sets
    x_train, y_train = data_split['x_train'], data_split['y_train']
    x_val, y_val = data_split['x_val'], data_split['y_val']
    
    # Load the config
    config = load_config()
    
    # Get the epochs and batch size
    epochs = config['model']['training_settings']['epochs']
    batch_size = config['model']['training_settings']['batch_size']

    # Form project name
    data_set = "five_years" if config['data']['dataset_index'] == 0 else "ten_years"
    model_type = config['model']['model_type']
    project_name = f'stock_prediction_{data_set}_{model_type}'

    # Tuner for LSTM
    if model_type == "LSTM":
        # Create the tuner
        tuner = RandomSearch(
            model,
            objective='val_loss',
            max_trials=config['LSTM']['tuner_params']['max_trials'],
            executions_per_trial=config['LSTM']['tuner_params']['executions_per_trial'],
            directory='./stock_prediction/models/tuning/',
            project_name=project_name
        )
    
        # Search for the best hyperparameters
        tuner.search(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size)
        
        # Get the best hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        
        # Print the best hyperparameters
        print(best_hps.values)
    
    # Tuner for SVR
    else:
        # Create the project directory
        tuner_dir = f'./stock_prediction/models/tuning/{project_name}'
        
        # Initialize the tuner
        tuner : RandomizedSearchCV = None
        
        # Check if the tuner exists if so load it
        if os.path.exists(f"{tuner_dir}/svr_tuner.pkl"):
            # Load the tuner
            tuner = joblib.load(f"{tuner_dir}/svr_tuner.pkl")
        
        # Use the loaded tuner if overwrite is False and tuner exists
        if not config['SVR']['tuner_params']['overwrite'] and tuner:
            # Get the best hyperparameters
            best_hps = tuner.best_params_
        # Otherwise, create the tuner
        else:            
            # Create the tuner
            tuner = RandomizedSearchCV(
                model,
                param_distributions=convert_svr_params(config['SVR']),
                n_iter=config['SVR']['tuner_params']['n_iter'],
                cv=config['SVR']['tuner_params']['cv'],
                n_jobs=config['SVR']['tuner_params']['n_jobs'],
                scoring='neg_mean_squared_error',
                refit=False,
                verbose=1,
                return_train_score=True
            )

            # Search for the best hyperparameters
            tuner.fit(x_train, y_train)

            # Get the best hyperparameters
            best_hps = tuner.best_params_
            
        
        # Create the directory
        os.makedirs(tuner_dir, exist_ok=True)
        
        # Save the tuner
        path = f"{tuner_dir}/svr_tuner.pkl"
        
        # Save the tuner
        joblib.dump(tuner, path)
        
        # Print the best hyperparameters
        print(best_hps)


    return best_hps
