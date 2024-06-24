'''
This package is used to load the configuration file for the project.
TODO: Might want to consider adding a class and storing the configurations as class variables.
'''

from .config_loader import load_config, fetch_active_features

__all__ = ['load_config', 'fetch_active_features']
