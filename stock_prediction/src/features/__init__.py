'''
This is the features package. It contains modules that are used to create features for the model.
'''

from .feature_gen import create_features, compute_macd, compute_rsi, compute_stochastic_oscillator, remove_unused_features

__all__ = ['create_features', 'compute_macd', 'compute_rsi', 'compute_stochastic_oscillator', 'remove_unused_features']
