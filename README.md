# Stock Prediction AI
---

## Project Description

This project aims to build an AI model to predict stock prices using the historical data of the stock market from Yahoo Finance. This is done by using YFinance to download the historical data of the stock market and then using the data to train the AI models. We will be using two different AI models. One being, the Long Short Term Memory (LSTM) model and the other being the Support Vector Machine (SVM) model. The LSTM model is a type of Recurrent Neural Network (RNN) that is capable of learning long-term dependencies. The SVM model is a supervised machine learning model that analyzes data for classification and regression analysis. The predicted stock prices will be compared with the actual stock prices to evaluate the performance of the AI models. 

### Project Structure
Here is an overview of the project structure:

#### `data/`
- **raw/**: This folder contains the raw data files that are used as input for the model.
- **processed/**: This folder is intended for storing processed data files that have been cleaned and preprocessed.

#### `models/`
- This file will store the trained model's checkpoints and the model's architecture.

#### `notebooks/`
- Jupyter notebooks for data exploration, visualization, and experimentation.

#### `src/`
- **config/**: Contains configuration files for the project.
- **data/**: Contains modules for data loading and preprocessing.
- **features/**: Contains modules for feature engineering.
- **models/**: Contains modules for model creation, training, and prediction.
- **utils/**: Utility functions that support various tasks in the project.

### `main.py`
- The main script to train the model and make predictions.

## Dataset
The dataset used in this project is the historical data of the stock market from [Yahoo Finance](https://finance.yahoo.com/).

### Features:
| Feature                | Description                       | Used |
|------------------------|-----------------------------------|------|
| Open                   | Opening price of the stock         | False  |
| High                   | Highest price of the stock         | False  |
| Low                    | Lowest price of the stock          | False  |
| Close                  | Closing price of the stock         | False  |
| Volume                 | Number of shares traded            | False  |
| Adj Close              | Adjusted closing price of the stock | False  |

## LSTM Model
### Hyperparameters:
| Hyperparameter          | Description                       | Default Value | Selected Value |
|-------------------------|-----------------------------------|---------------|----------------|
| Number of LSTM layers   | Number of layers in LSTM model    | 2             |                |
| Number of neurons in each LSTM layer | Number of neurons in each LSTM layer | 128           |                |
| Number of epochs        | Number of training epochs          | 100           |                |
| Batch size              | Number of samples per batch        | 32            |                |
| Learning rate           | Rate at which the model learns     | 0.001         |                |
| Dropout rate            | Rate at which neurons are randomly dropped out | 0.2           |                |
| Time steps              | Number of time steps in LSTM model | 60            |                |
| Loss function           | Function used to calculate the difference between predicted and actual values | Mean Squared Error |                |
| Activation function     | Function used to introduce non-linearity in the model | ReLU          |                |
| Number of features      | Number of input features           | 6             |                |
| Length of historical data | Number of past time steps used as input for prediction | 60            |

## SVR Model
### Hyperparameters:
| Parameter              | Description                       | Default Value | Selected Value |
|------------------------|-----------------------------------|---------------|----------------|
| Kernel                 | Type of kernel function used in the SVM model | "rbf"         |                |
| C                      | Penalty parameter of the error term in the SVM model | 1.0           |                |
| Gamma                  | Kernel coefficient for 'rbf', 'poly', and 'sigmoid' kernels | "scale"       |                |
| Degree                 | Degree of the polynomial kernel function | 3             |                |
| Coef0                  | Independent term in the polynomial kernel function | 0.0           |                |
| Epsilon                | Epsilon in the epsilon-SVR model | 0.1           |                |
| Number of features     | Number of input features           | 6             |                |
| Length of historical data | Number of past time steps used as input for prediction | 60            |                
  

## Main Research Question
*Can we predict the stock prices of the stock market using AI models? If so, which AI model is more accurate in predicting the stock prices?*

### Sub-Research Questions
1. How the length of the historical data affects the accuracy of the AI models?
2. Which parameters of the AI models affect the accuracy of the AI models?
3. How the number of features affects the accuracy of the AI models?

# Installation
! Supported python version: 3.9.x

1) To install the required libraries, run the following command:
```pip install -r requirements.txt```
2) To run the code, run the following command:
```python main.py```
