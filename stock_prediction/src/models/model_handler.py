# pylint: disable= import-error, C0301, E0611,

"""
This package handles the model creation, training
"""

# Default Libraries
import os
from typing import Union
import joblib

# Third Party Libraries
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
import tensorflow as tf  # type: ignore
from kerastuner import HyperModel
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Custom Libraries
from stock_prediction.src.config_loader import load_config

class LSTMModelHandler(HyperModel):
    """
    This class is used to create the LSTM, and tune the hyperparameters of the models.
    """

    def __init__(self, input_shape: tuple):
        # Set the input shape
        self.input_shape = input_shape
        
        print(f'Input shape: {input_shape}')

        # Load the config
        self.config = load_config()

        # Get the LSTM settings
        self.lstm_settings = self.config["LSTM"]

        # Device
        self.device = '/GPU:0' if tf.config.experimental.list_physical_devices('GPU') else '/CPU:0'

        self.hyperparameters = {}

        # Call the parent class constructor
        super().__init__()

    def _set_lstm_hyperparameters(self, hp):
        '''
        Set the hyperparameters for the LSTM model
        '''

        # Hyperparameters (fetched from the config.json)
        self.hyperparameters["n_layers"] = hp.Int(
            "n_layers",
            min_value=self.lstm_settings["n_layers"]["min"],
            max_value=self.lstm_settings["n_layers"]["max"],
            step=self.lstm_settings["n_layers"]["step"],
        )
        self.hyperparameters["n_nodes"] = hp.Int(
            "n_nodes",
            min_value=self.lstm_settings["n_nodes"]["min"],
            max_value=self.lstm_settings["n_nodes"]["max"],
            step=self.lstm_settings["n_nodes"]["step"],
        )
        self.hyperparameters["dropout"] = hp.Float(
            "dropout",
            min_value=self.lstm_settings["dropout"]["min"],
            max_value=self.lstm_settings["dropout"]["max"],
            step=self.lstm_settings["dropout"]["step"],
        )
        self.hyperparameters["dense_nodes"] = hp.Int(
            "dense_nodes",
            min_value=self.lstm_settings["dense_nodes"]["min"],
            max_value=self.lstm_settings["dense_nodes"]["max"],
            step=self.lstm_settings["dense_nodes"]["step"],
        )
        self.hyperparameters["learning_rate"] = hp.Choice(
            "learning_rate", values=self.lstm_settings["learning_rate"]
        )
        self.hyperparameters["loss"] = hp.Choice(
            "loss", values=self.lstm_settings["loss"]
        )

    def _load_hyperparameters(self, hp):
        """
        Load the hyperparameters from the config file
        """
        
        self._set_lstm_hyperparameters(hp)

    def build(self, hp):
        """
        Builds the model based on the model type. This is a built-in function in the HyperModel class.
        This function will be called by the tuner to build the model. Not while manually building the model.
        Hence, the hyperparameters are loaded in this function from the config file. set_hyperparameters() is not called.
        """

        # Load the hyperparameters
        self._load_hyperparameters(hp)

        return self.create_lstm_model()


    def create_lstm_model(self):
        """
        Creates the LSTM model with the hyperparameters
        """

        with tf.device(self.device):
            # Initialize a sequential model, which is a linear stack of layers
            model = Sequential()
            # Add input layer
            model.add(Input(shape=self.input_shape))

            # Add layers
            model.add(LSTM(self.hyperparameters["n_nodes"], return_sequences=True))
            model.add(Dropout(self.hyperparameters["dropout"]))

            for _ in range(self.hyperparameters["n_layers"] - 1):
                model.add(LSTM(self.hyperparameters["n_nodes"], return_sequences=True))
                model.add(Dropout(self.hyperparameters["dropout"]))

            model.add(Dense(self.hyperparameters["dense_nodes"]))
            model.add(BatchNormalization())
            model.add(Dense(1))

            # Optimizer
            optimizer = Adam(learning_rate=self.hyperparameters["learning_rate"])
            # Compile the model
            model.compile(loss=self.hyperparameters["loss"], optimizer=optimizer)

            # Return the model
            return model


def train_model(data_split: dict, model: Union[LSTMModelHandler, SVR], best_hp: dict = None):
    """
    Main function to train the LSTM model

    args:
        data_split: the split data (dict)
            format: {'x_train': np.ndarray, 'y_train': np.ndarray, 'x_val': np.ndarray, 'y_val': np.ndarray}
        model: the model handler instance
    """
    
    # Get the training, validation, and test sets
    x_train, y_train = data_split["x_train"], data_split["y_train"]
    x_val, y_val = data_split["x_val"], data_split["y_val"]
    
    # Load the config
    config = load_config()
    
    # Build the model
    if config['model']['model_type'] == "LSTM":
        # Type hinting
        model: LSTMModelHandler = model
        
        model = model.build(best_hp)
    
        # Early stopping
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=config['model']['training_settings']["early_stopping"]["patience"],
            restore_best_weights=True,
        )
        
        # Train the model
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=config['model']['training_settings']["epochs"],
            batch_size=config['model']['training_settings']["batch_size"],
            callbacks=[early_stopping],
        )
        
        # Save the model
        model.save(f"./stock_prediction/models/{config['model']['model_type']}_model")
    else:
        # Re initialize the model with the best hyperparameters
        model = SVR(**best_hp)
        
        # Train the model
        model.fit(x_train, y_train)
        history = None
        
        # Create dir if it does not exist
        model_dir = f"./stock_prediction/models/{config['model']['model_type']}_model/"
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the model
        joblib.dump(model, f"{model_dir}svr_model.pkl")
        
    # Return the history and model
    return history, model

def evaluate_model(model : Union[LSTMModelHandler, SVR], data_split : dict):
    """
    This function is used to evaluate the model.

    Parameters:
        model: Model
            The model to evaluate.
        data_split: dict
            The data to use for evaluating the model.
            format: {'x_train': np.ndarray, 'y_train': np.ndarray, 'x_val': np.ndarray, 'y_val': np.ndarray}
    """
    
    # Get the training, validation, and test sets
    x_test, y_test = data_split["x_test"], data_split["y_test"]
    
    # Load the config
    config = load_config()
    
    if config['model']['model_type'] == "LSTM":
        # Evaluate the model
        loss = model.evaluate(x_test, y_test)
    else:
        loss = mean_squared_error(y_test, model.predict(x_test))

    # Return the loss (rounded to 6 decimal places)
    return format(loss, ".6f")

def analyze_model(model: Union[LSTMModelHandler, SVR], data_split: dict, scalers : dict):
    """
    This function is used to analyze the model.

    Parameters:
        model: Model
            The model to analyze.
        data_split: dict
            The data to use for analyzing the model.
            format: {'x_train': np.ndarray, 'y_train': np.ndarray, 'x_val': np.ndarray, 'y_val': np.ndarray}
    """
    # Get the training, validation, and test sets
    x_test, y_test = data_split["x_test"], data_split["y_test"]

    # Fetch scalers
    target_scaler = scalers["target_scaler"]

    # Load the config
    config = load_config()

    # Predict the values
    predictions = model.predict(x_test)
    
    print(f'First few processed predictions: {predictions[:20]}')
    
    print(predictions.shape)
    
    if config['model']['model_type'] == "SVR":
        # Inverse transform the predictions
        predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        # Inverse transform the actual values
        y_test = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    else:
        # LSTM predictions typically have the shape (n_samples, n_timesteps, n_features)
        # If necessary, reduce the dimensions to (n_samples, n_features)
        if len(predictions.shape) == 3:
            predictions = predictions[:, -1, 0]  # Assuming we need the last time step's prediction
        elif len(predictions.shape) == 2:
            predictions = predictions[:, 0]  # Assuming we have (n_samples, 1) shape

        # Inverse transform the predictions
        predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        # Inverse transform the actual values
        y_test = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
    # Plot the predictions
    plt.figure(figsize=(14, 7))
    plt.plot(y_test, label="Actual")
    plt.plot(predictions, label="Predicted")
    plt.xlabel('Samples')
    plt.ylabel('Values')
    plt.legend()
    plt.show()
    
    # Return the predictions
    return predictions
