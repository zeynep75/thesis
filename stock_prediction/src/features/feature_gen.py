# pylint: disable= C0301, import-error

'''
This module handles the feature generation for the project. It contains functions to generate features for the data.
'''

# Third-party imports
import pandas as ps

# Custom imports
from stock_prediction.src.config_loader import fetch_active_features

def create_features(df : ps.DataFrame):
    '''
    Creates features for a stock/index based on historical price data.
    '''
    
    # Open-Close, which is the difference between the opening and closing prices for the day indicates the price movement of the stock/index
    df['Open-Close'] = df['Open'] - df['Close']
    
    # High-Low, which is the difference between the high and low prices for the day indicates the price volatility of the stock/index
    df['High-Low'] = df['High'] - df['Low']

    # Moving Averages, which means the average of the last n days, which smoothens the price data to identify trends (close price is used here)
    df['7-day MA'] = df['Close'].rolling(window=7).mean()
    df['21-day MA'] = df['Close'].rolling(window=21).mean()
    
    # Stochastic Oscillator, which measures where the close is in relation to the recent high-low range
    df['SO'] = compute_stochastic_oscillator(df, 14)
    
    # Relative Strength Index (RSI), which is a momentum oscillator 
    # that measures the speed and change of price movements
    df['RSI'] = compute_rsi(df['Close'], 14)
    
    # Moving Average Convergence Divergence (MACD), which is a trend-following 
    # momentum indicator that shows the relationship between two moving averages 
    # of a security’s price
    df['MACD'] = compute_macd(df['Close'])
    
    return df

def remove_unused_features(df : ps.DataFrame):
    '''
    Removes unused features from the dataframe.
    '''
    
    # Fetch the active features
    active_features = fetch_active_features()
    # Drop the unused features
    df.drop(columns=[col for col in df.columns if col not in active_features], inplace=True)
    return df

def compute_rsi(series, window):
    '''
    Computes the Relative Strength Index (RSI) for a given series.
    '''
    
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, slow=26, fast=12):
    '''
    Computes the Moving Average Convergence Divergence (MACD) for a given series.
    '''
    
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    return macd

def compute_stochastic_oscillator(df, window):
    '''
    Computes the Stochastic Oscillator for a given dataframe.
    '''
    
    low_min = df['Low'].rolling(window=window).min()
    high_max = df['High'].rolling(window=window).max()
    stoch_oscillator = ((df['Close'] - low_min) / (high_max - low_min)) * 100
    return stoch_oscillator
