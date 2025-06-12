#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-06-01 13:20:00 (ywatanabe)"
# File: ./tests/mngs/ai/optim/test__get_set.py

"""Tests for mngs.ai.optim._get_set module (deprecated functions)."""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from mngs.ai.optim import get, set


class TestGetSet:
    """Test suite for deprecated get/set functions."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple neural network model."""
        return nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )

    @pytest.fixture
    def model_list(self):
        """Create a list of models."""
        return [
            nn.Linear(10, 5),
            nn.Linear(5, 1)
        ]

    def test_get_adam(self):
        """Test getting Adam optimizer."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            optimizer_class = get('adam')
            
        assert optimizer_class == optim.Adam
        assert issubclass(optimizer_class, optim.Optimizer)

    def test_get_sgd(self):
        """Test getting SGD optimizer."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            optimizer_class = get('sgd')
            
        assert optimizer_class == optim.SGD
        assert issubclass(optimizer_class, optim.Optimizer)

    def test_get_rmsprop(self):
        """Test getting RMSprop optimizer."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            optimizer_class = get('rmsprop')
            
        assert optimizer_class == optim.RMSprop
        assert issubclass(optimizer_class, optim.Optimizer)

    def test_get_invalid_optimizer(self):
        """Test getting invalid optimizer raises error."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match="Unknown optimizer"):
                get('invalid_optimizer')

    def test_set_single_model(self, simple_model):
        """Test setting optimizer for a single model."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            optimizer = set(simple_model, 'adam', 0.001)
            
        assert isinstance(optimizer, optim.Adam)
        assert optimizer.defaults['lr'] == 0.001
        
        # Check that all model parameters are in optimizer
        model_params = list(simple_model.parameters())
        optim_params = []
        for group in optimizer.param_groups:
            optim_params.extend(group['params'])
        assert len(model_params) == len(optim_params)

    def test_set_model_list(self, model_list):
        """Test setting optimizer for a list of models."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            optimizer = set(model_list, 'sgd', 0.01)
            
        assert isinstance(optimizer, optim.SGD)
        assert optimizer.defaults['lr'] == 0.01
        
        # Check that all parameters from all models are included
        total_params = sum(len(list(model.parameters())) for model in model_list)
        optim_params = []
        for group in optimizer.param_groups:
            optim_params.extend(group['params'])
        assert len(optim_params) == total_params

    def test_set_different_learning_rates(self, simple_model):
        """Test setting different learning rates."""
        learning_rates = [1e-4, 1e-3, 1e-2]
        
        for lr in learning_rates:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                optimizer = set(simple_model, 'adam', lr)
                
            assert optimizer.defaults['lr'] == lr

    def test_deprecation_warning_get(self):
        """Test that get function issues deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            get('adam')
            
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "mngs.ai.optim.get is deprecated" in str(w[0].message)
        assert "get_optimizer" in str(w[0].message)

    def test_deprecation_warning_set(self, simple_model):
        """Test that set function issues deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            set(simple_model, 'adam', 0.001)
            
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "mngs.ai.optim.set is deprecated" in str(w[0].message)
        assert "set_optimizer" in str(w[0].message)

    def test_get_ranger_conditional(self):
        """Test getting Ranger optimizer based on availability."""
        from mngs.ai.optim import RANGER_AVAILABLE
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            
            if RANGER_AVAILABLE:
                # Should return Ranger class
                optimizer_class = get('ranger')
                assert optimizer_class is not None
                assert optimizer_class.__name__ in ['Ranger', 'Ranger21']
            else:
                # Should raise ImportError
                with pytest.raises(ImportError, match="Ranger optimizer not available"):
                    get('ranger')

    def test_set_with_no_parameters(self):
        """Test setting optimizer with model that has no parameters."""
        # Create a model with no learnable parameters
        class NoParamModel(nn.Module):
            def forward(self, x):
                return x
                
        model = NoParamModel()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            optimizer = set(model, 'adam', 0.001)
            
        assert isinstance(optimizer, optim.Adam)
        # Should have no parameters
        assert all(len(group['params']) == 0 for group in optimizer.param_groups)

    @pytest.mark.parametrize("optim_name", ['adam', 'sgd', 'rmsprop'])
    def test_set_all_standard_optimizers(self, simple_model, optim_name):
        """Test setting all standard optimizers."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            optimizer = set(simple_model, optim_name, 0.001)
            
        expected_class = {
            'adam': optim.Adam,
            'sgd': optim.SGD,
            'rmsprop': optim.RMSprop
        }[optim_name]
        
        assert isinstance(optimizer, expected_class)

    def test_function_imports_from_module(self):
        """Test that functions can be imported directly from _get_set."""
        from mngs.ai.optim._get_set import get as get_func
        from mngs.ai.optim._get_set import set as set_func
        
        assert callable(get_func)
        assert callable(set_func)
        
        # They should be the same as the public API
        assert get_func is get
        assert set_func is set


if __name__ == "__main__":
<<<<<<< HEAD
    pytest.main([__file__, "-v"])
=======
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/ai/optim/_get_set.py
# --------------------------------------------------------------------------------
# # import torch.nn as nn
# import torch.optim as optim
# from .Ranger_Deep_Learning_Optimizer.ranger.ranger2020 import Ranger
# 
# 
# # def set_an_optim(models, optim_str, lr):
# def set(models, optim_str, lr):    
#     """Sets an optimizer to models"""
#     if not isinstance(models, list):
#         models = [models]
#     learnable_params = []
#     for m in models:
#         learnable_params += list(m.parameters())
#     # optim = mngs.ml.switch_optim(optim_str)
#     optim = get(optim_str)    
#     return optim(learnable_params, lr)
# 
# def get(optim_str):
#     optims_dict = {
#         "adam": optim.Adam,
#         "ranger": Ranger,
#         "rmsprop": optim.RMSprop,
#         "sgd": optim.SGD
#         }
#     return optims_dict[optim_str]

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/ai/optim/_get_set.py
# --------------------------------------------------------------------------------
>>>>>>> origin/main
