'''
This package is used to load, clean, and preprocess the data for the project.
'''

from .data_handler import load_data, handle_data
from.preprocessing import preprocess_data, normalize_data, prepare_data

__all__ = ['load_data', 'preprocess_data', 'normalize_data', 'prepare_data', 'handle_data']
