"""
This module is the entry point for the project. It is used to run the project 
and execute the main code.
"""

# Third-party libraries
import tensorflow as tf # type: ignore
from sklearn.svm import SVR


# Custom Imports
from stock_prediction.src import (
    handle_data,
    load_data,
    tune_hyperparameters,
    train_model,
    evaluate_model,
    LSTMModelHandler,
    load_config,
    analyze_model
)

def main():
    """
    This function is the entry point for the project. It is used to run the project and
    execute the main code.
    """

    # Load the data
    data = load_data()
    
    # Load config
    config = load_config()

    # Check if the data was loaded
    if data is None:
        return

    # Handle the data(process) and split it into training, validation, and test sets
    data_split, scaler = handle_data(data)

    # Check if the data was split
    if data_split is None:
        return

    if config['model']['model_type'] == 'LSTM':
        # Initialize the model handler
        model_handler = LSTMModelHandler(
            input_shape=(data_split["x_train"].shape[1], data_split["x_train"].shape[2])
        )
    else:
        model_handler = SVR()

    # Tune the hyperparameters and get the best hyperparameters
    best_hp = tune_hyperparameters(model_handler, data_split)

    # Train the model
    history, model = train_model(data_split, model_handler, best_hp)
    
    # Evaluate the modelsc
    eval_loss = evaluate_model(model, data_split)

    print("Training completed.")
    print(f"Evaluation Loss: {eval_loss}")
    
    # Analyze the model
    preds = analyze_model(model, data_split, scaler)

if __name__ == "__main__":
    print('Number of GPUs Available: ', len(tf.config.experimental.list_physical_devices('GPU')))

    main()
