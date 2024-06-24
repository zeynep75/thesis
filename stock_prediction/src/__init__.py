# pylint: disable= C0301

"""
This is the source package for the project. It contains most of the primary code for the project.
"""

from .config_loader import load_config, fetch_active_features
from .data import load_data, preprocess_data, normalize_data, prepare_data, handle_data
from .models import LSTMModelHandler, train_model, tune_hyperparameters, evaluate_model, analyze_model
from .features import (
    create_features,
    compute_macd,
    compute_rsi,
    compute_stochastic_oscillator,
    remove_unused_features,
)

__all__ = [
    "load_config",
    "load_data",
    "preprocess_data",
    "normalize_data",
    "prepare_data",
    "LSTMModelHandler",
    "train_model",
    "create_features",
    "compute_macd",
    "compute_rsi",
    "compute_stochastic_oscillator",
    "fetch_active_features",
    "handle_data",
    "remove_unused_features",
    "tune_hyperparameters",
    "evaluate_model",
    "analyze_model"
]
