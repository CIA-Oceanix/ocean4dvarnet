"""
Ocean4DVarNet: A framework for solving inverse problems in data assimilation 
using deep learning and PyTorch.

This package provides modules for:
- Models: Definition of neural network architectures and solvers.
- Data: Data loading and preprocessing utilities.
- Training: Training and evaluation pipelines using PyTorch Lightning.
- Processing: Post-processing utilities for model outputs.
- Plotting: Visualization tools for results and diagnostics.
- Evaluation: Metrics and evaluation functions for model performance.
- Utilities: General-purpose helper functions.

Modules:
    models: Contains model architectures and solvers.
    data: Handles data loading and preprocessing.
    train: Implements training and testing pipelines.
    process: Provides post-processing utilities.
    plot: Includes visualization tools.
    evaluate: Defines evaluation metrics and functions.
    utils: Contains general-purpose utility functions.
"""
from . import models
from . import data
from . import train
from . import process
from . import plot
from . import evaluate
from . import utils

# Add function in exported function list
__all__ = [
    'models',
    'data',
    'train',
    'process',
    'plot',
    'evaluate',
    'utils'
]
