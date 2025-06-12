#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive test suite for mngs.ai.utils module initialization.

This test module verifies:
- Module imports and structure
- Public API availability
- Import error handling
- Module attributes
"""

import sys
import importlib
import pytest
import types


class TestAIUtilsInit:
    """Test cases for AI utils module initialization."""
    
    def test_module_import(self):
        """Test that the module can be imported successfully."""
        import mngs.ai.utils
        assert mngs.ai.utils is not None
        assert isinstance(mngs.ai.utils, types.ModuleType)
    
    def test_grid_search_submodule_available(self):
        """Test that grid_search submodule is available."""
        import mngs.ai.utils
        assert hasattr(mngs.ai.utils, 'grid_search')
        assert isinstance(mngs.ai.utils.grid_search, types.ModuleType)
    
    def test_check_params_available(self):
        """Test that check_params function is available."""
        from mngs.ai.utils import check_params
        assert callable(check_params)
    
    def test_default_dataset_available(self):
        """Test that DefaultDataset class is available."""
        from mngs.ai.utils import DefaultDataset
        assert isinstance(DefaultDataset, type)
    
    def test_label_encoder_available(self):
        """Test that LabelEncoder class is available."""
        from mngs.ai.utils import LabelEncoder
        assert isinstance(LabelEncoder, type)
    
    def test_format_samples_for_sktime_available(self):
        """Test that format_samples_for_sktime function is available."""
        from mngs.ai.utils import format_samples_for_sktime
        assert callable(format_samples_for_sktime)
    
    def test_merge_labels_available(self):
        """Test that merge_labels function is available."""
        from mngs.ai.utils import merge_labels
        assert callable(merge_labels)
    
    def test_sliding_window_data_augmentation_available(self):
        """Test that sliding_window_data_augmentation function is available."""
        from mngs.ai.utils import sliding_window_data_augmentation
        assert callable(sliding_window_data_augmentation)
    
    def test_under_sample_available(self):
        """Test that under_sample function is available."""
        from mngs.ai.utils import under_sample
        assert callable(under_sample)
    
    def test_verify_n_gpus_available(self):
        """Test that verify_n_gpus function is available."""
        from mngs.ai.utils import verify_n_gpus
        assert callable(verify_n_gpus)
    
    def test_module_all_attribute(self):
        """Test module __all__ attribute if defined."""
        import mngs.ai.utils
        # __all__ might not be defined, which is okay
        if hasattr(mngs.ai.utils, '__all__'):
            assert isinstance(mngs.ai.utils.__all__, list)
            assert len(mngs.ai.utils.__all__) > 0
    
    def test_grid_search_functions(self):
        """Test that grid_search module has expected functions."""
        from mngs.ai.utils import grid_search
        assert hasattr(grid_search, 'yield_grids')
        assert hasattr(grid_search, 'count_grids')
        assert callable(grid_search.yield_grids)
        assert callable(grid_search.count_grids)
    
    def test_no_circular_imports(self):
        """Test that there are no circular import issues."""
        # Force reload to check for circular imports
        import mngs.ai.utils
        importlib.reload(mngs.ai.utils)
        # If we get here without errors, no circular imports
        assert True
    
    def test_module_path(self):
        """Test that module has correct path attributes."""
        import mngs.ai.utils
        assert hasattr(mngs.ai.utils, '__file__')
        assert hasattr(mngs.ai.utils, '__name__')
        assert mngs.ai.utils.__name__ == 'mngs.ai.utils'
    
    def test_import_from_syntax(self):
        """Test various import from syntaxes."""
        # Test individual imports
        from mngs.ai.utils import check_params, LabelEncoder
        assert check_params is not None
        assert LabelEncoder is not None
        
        # Test wildcard import behavior
        namespace = {}
        exec("from mngs.ai.utils import *", namespace)
        # Should have imported public functions/classes
        assert 'check_params' in namespace or '__all__' not in dir(importlib.import_module('mngs.ai.utils'))
    
    def test_submodule_independence(self):
        """Test that submodules can be imported independently."""
        # Clear any cached imports
        for key in list(sys.modules.keys()):
            if key.startswith('mngs.ai.utils'):
                del sys.modules[key]
        
        # Import just grid_search
        from mngs.ai.utils import grid_search
        assert grid_search is not None
    
    def test_reloading_module(self):
        """Test that the module can be reloaded without issues."""
        import mngs.ai.utils
        original_id = id(mngs.ai.utils)
        
        # Reload the module
        importlib.reload(mngs.ai.utils)
        
        # Module should still be functional
        assert hasattr(mngs.ai.utils, 'check_params')
        assert hasattr(mngs.ai.utils, 'LabelEncoder')
    
    def test_parent_module_access(self):
        """Test access to utils module through parent ai module."""
        import mngs.ai
        assert hasattr(mngs.ai, 'utils')
        assert hasattr(mngs.ai.utils, 'check_params')


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/ai/utils/__init__.py
# --------------------------------------------------------------------------------
# # from ._yield_grid_params import yield_grid_params
# from . import grid_search
# from ._check_params import check_params
# from ._DefaultDataset import DefaultDataset
# from ._format_samples_for_sktime import format_samples_for_sktime
# from ._LabelEncoder import LabelEncoder
# from ._merge_labels import merge_labels
# from ._sliding_window_data_augmentation import sliding_window_data_augmentation
# from ._under_sample import under_sample
# from ._verify_n_gpus import verify_n_gpus

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/ai/utils/__init__.py
# --------------------------------------------------------------------------------
