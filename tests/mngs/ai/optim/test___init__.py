#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-06-01 13:15:00 (ywatanabe)"
# File: ./tests/mngs/ai/optim/test___init__.py

"""Tests for mngs.ai.optim module initialization."""

import pytest
import torch
import torch.nn as nn
import warnings
import mngs


class TestOptimInit:
    """Test suite for mngs.ai.optim module initialization."""

    def test_module_import(self):
        """Test that the module can be imported."""
        assert hasattr(mngs.ai, 'optim')
        
    def test_function_imports(self):
        """Test that key functions are imported."""
        # New API
        assert hasattr(mngs.ai.optim, 'get_optimizer')
        assert hasattr(mngs.ai.optim, 'set_optimizer')
        
        # Legacy API (deprecated but should still exist)
        assert hasattr(mngs.ai.optim, 'get')
        assert hasattr(mngs.ai.optim, 'set')
        
        # Ranger availability flag
        assert hasattr(mngs.ai.optim, 'RANGER_AVAILABLE')
        
    def test_function_callable(self):
        """Test that imported functions are callable."""
        assert callable(mngs.ai.optim.get_optimizer)
        assert callable(mngs.ai.optim.set_optimizer)
        assert callable(mngs.ai.optim.get)
        assert callable(mngs.ai.optim.set)
        
    def test_ranger_availability_flag(self):
        """Test that RANGER_AVAILABLE is a boolean."""
        assert isinstance(mngs.ai.optim.RANGER_AVAILABLE, bool)
        
    def test_import_from_module(self):
        """Test direct imports from the module."""
        from mngs.ai.optim import get_optimizer, set_optimizer, get, set, RANGER_AVAILABLE
        
        assert callable(get_optimizer)
        assert callable(set_optimizer)
        assert callable(get)
        assert callable(set)
        assert isinstance(RANGER_AVAILABLE, bool)
        
    def test_module_all_attribute(self):
        """Test that __all__ is properly defined."""
        expected_exports = ["get_optimizer", "set_optimizer", "get", "set", "RANGER_AVAILABLE"]
        
        if hasattr(mngs.ai.optim, '__all__'):
            module_all = mngs.ai.optim.__all__
            for export in expected_exports:
                assert export in module_all
                
    def test_no_private_exports(self):
        """Test that private implementation details are not exposed."""
        public_attrs = [attr for attr in dir(mngs.ai.optim) 
                       if not attr.startswith('_')]
        
        # These should be the only public exports
        expected_public = {'get_optimizer', 'set_optimizer', 'get', 'set', 
                          'RANGER_AVAILABLE', 'MIGRATION'}
        
        # Allow standard module attributes
        allowed_attrs = {'__name__', '__doc__', '__package__', '__loader__', 
                        '__spec__', '__file__', '__cached__', '__builtins__',
                        '__all__', '__path__'}
        
        actual_public = set(public_attrs) - allowed_attrs
        
        # Should not have unexpected exports
        unexpected = actual_public - expected_public
        assert len(unexpected) == 0, f"Unexpected exports: {unexpected}"
        
    def test_deprecation_warnings(self):
        """Test that deprecated functions issue warnings."""
        # Test deprecated 'get' function
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mngs.ai.optim.get('adam')
            
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message)
            assert "get_optimizer" in str(w[0].message)
            
        # Test deprecated 'set' function
        model = nn.Linear(10, 1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mngs.ai.optim.set(model, 'adam', 0.001)
            
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message)
            assert "set_optimizer" in str(w[0].message)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])