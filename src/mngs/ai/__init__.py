#!/usr/bin/env python3
"""AI module for MNGS - Machine Learning and AI utilities."""

# Import main functionality from submodules
from .genai import GenAI  # Main GenAI factory

# Training utilities
from .training import EarlyStopping, LearningCurveLogger

# Classification utilities  
from .classification import ClassificationReporter, ClassifierServer

# Import submodules for namespace access
from . import (
    act,
    classification,
    clustering,
    feature_extraction,
    genai,
    layer,
    loss,
    metrics,
    optim,
    plt,
    sampling,
    sklearn,
    training,
    utils,
)

# For backward compatibility, expose some commonly used items at top level
from .optim import get_optimizer, set_optimizer

__all__ = [
    # Main factory
    'GenAI',
    # Training
    'EarlyStopping',
    'LearningCurveLogger',
    # Classification
    'ClassificationReporter', 
    'ClassifierServer',
    # Submodules
    'act',
    'classification',
    'clustering',
    'feature_extraction',
    'genai',
    'layer',
    'loss',
    'metrics',
    'optim',
    'plt',
    'sampling',
    'sklearn',
    'training',
    'utils',
    # Optimizer functions
    'get_optimizer',
    'set_optimizer',
]