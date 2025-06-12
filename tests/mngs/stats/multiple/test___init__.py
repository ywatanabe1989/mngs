#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./tests/mngs/stats/multiple/test___init__.py

import pytest
import numpy as np
import pandas as pd
import mngs


class TestImports:
    def test_import_main(self):
        import mngs

    def test_import_submodule(self):
        import mngs.stats

    def test_import_multiple_submodule(self):
        import mngs.stats.multiple


class TestMultipleSubmodule:
    """Test the multiple corrections submodule structure and imports."""
    
    def test_submodule_has_functions(self):
        """Test that multiple submodule has expected functions."""
        assert hasattr(mngs.stats.multiple, 'bonferroni_correction')
        assert hasattr(mngs.stats.multiple, 'fdr_correction')
        assert hasattr(mngs.stats.multiple, 'multicompair')
    
    def test_module_structure(self):
        """Test the module structure and organization."""
        import mngs.stats.multiple
        
        # Check for implementation modules
        assert hasattr(mngs.stats.multiple, '_bonferroni_correction')
        assert hasattr(mngs.stats.multiple, '_fdr_correction')
        assert hasattr(mngs.stats.multiple, '_multicompair')
    
    def test_accessible_from_stats(self):
        """Test that multiple correction functions are accessible from stats module."""
        # These should be available directly from stats module
        assert hasattr(mngs.stats, 'bonferroni_correction')
        assert hasattr(mngs.stats, 'fdr_correction')
        assert hasattr(mngs.stats, 'multicompair')


class TestModuleFunctionality:
    """Test basic functionality of multiple testing correction functions."""
    
    def test_bonferroni_basic(self):
        """Test basic Bonferroni correction functionality."""
        p_values = np.array([0.01, 0.04, 0.03])
        
        # Test through stats module
        corrected = mngs.stats.bonferroni_correction(p_values)
        assert isinstance(corrected, np.ndarray)
        assert len(corrected) == len(p_values)
        assert np.all(corrected >= p_values)
    
    def test_fdr_basic(self):
        """Test basic FDR correction functionality."""
        p_values = np.array([0.01, 0.04, 0.03])
        
        # Test through stats module
        corrected = mngs.stats.fdr_correction(p_values)
        assert isinstance(corrected, np.ndarray)
        assert len(corrected) == len(p_values)
        assert np.all(corrected >= p_values)
    
    def test_multicompair_basic(self):
        """Test basic multiple comparison functionality."""
        # Create sample groups
        group1 = np.random.randn(20)
        group2 = np.random.randn(20) + 0.5
        group3 = np.random.randn(20) + 1.0
        
        groups = [group1, group2, group3]
        
        # Test through stats module
        result = mngs.stats.multicompair(groups)
        assert isinstance(result, dict)
        assert 'summary' in result
        assert 'p_values' in result
        assert 'test_statistic' in result


class TestIntegration:
    """Integration tests for multiple testing correction workflow."""
    
    def test_correction_comparison(self):
        """Test that different correction methods give expected relationships."""
        p_values = np.array([0.001, 0.01, 0.02, 0.03, 0.04])
        
        # Apply different corrections
        bonf_corrected = mngs.stats.bonferroni_correction(p_values)
        fdr_corrected = mngs.stats.fdr_correction(p_values)
        
        # Bonferroni should be more conservative than FDR
        assert np.all(bonf_corrected >= fdr_corrected)
        
        # Both should be >= original p-values
        assert np.all(bonf_corrected >= p_values)
        assert np.all(fdr_corrected >= p_values)
    
    def test_dataframe_workflow(self):
        """Test multiple testing corrections in DataFrame workflow."""
        # Create test results DataFrame
        df = pd.DataFrame({
            'test_name': ['test1', 'test2', 'test3', 'test4', 'test5'],
            'p_value': [0.001, 0.01, 0.02, 0.03, 0.04],
            'effect_size': [0.8, 0.5, 0.3, 0.2, 0.1]
        })
        
        # Apply FDR correction (DataFrame version)
        corrected_df = mngs.stats.multiple._fdr_correction.fdr_correction(df)
        
        # Check that FDR columns were added
        assert 'p_value_fdr' in corrected_df.columns
        assert 'p_value_fdr_stars' in corrected_df.columns
        
        # Original columns should be preserved
        assert 'test_name' in corrected_df.columns
        assert 'effect_size' in corrected_df.columns
    
    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        # Empty array
        empty_array = np.array([])
        
        # Bonferroni with empty input
        bonf_result = mngs.stats.bonferroni_correction(empty_array)
        assert len(bonf_result) == 0
        
        # FDR with empty input
        fdr_result = mngs.stats.fdr_correction(empty_array)
        assert len(fdr_result) == 0
        
        # Empty DataFrame
        empty_df = pd.DataFrame({'p_value': []})
        corrected_df = mngs.stats.multiple._fdr_correction.fdr_correction(empty_df)
        assert len(corrected_df) == 0
        assert 'p_value_fdr' in corrected_df.columns


# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/stats/multiple/__init__.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
<<<<<<< HEAD
# # Time-stamp: "2024-11-05 06:22:45 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/stats/multiple/__init__.py
#
# """Module initialization for multiple comparison corrections."""
#
# from ._bonferroni_correction import bonferroni_correction, bonferroni_correction_torch
# from ._fdr_correction import fdr_correction
# from ._multicompair import multicompair
#
# __all__ = [
#     "bonferroni_correction",
#     "bonferroni_correction_torch", 
#     "fdr_correction",
#     "multicompair",
# ]
#
# # EOF
=======
# # Time-stamp: "2024-11-03 03:22:30 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/stats/desc/__init__.py
# 
# import os
# import importlib
# import inspect
# 
# # Get the current directory
# current_dir = os.path.dirname(__file__)
# 
# # Iterate through all Python files in the current directory
# for filename in os.listdir(current_dir):
#     if filename.endswith(".py") and not filename.startswith("__"):
#         module_name = filename[:-3]  # Remove .py extension
#         module = importlib.import_module(f".{module_name}", package=__name__)
# 
#         # Import only functions and classes from the module
#         for name, obj in inspect.getmembers(module):
#             if inspect.isfunction(obj) or inspect.isclass(obj):
#                 if not name.startswith("_"):
#                     globals()[name] = obj
# 
# # Clean up temporary variables
# del os, importlib, inspect, current_dir, filename, module_name, module, name, obj
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/stats/multiple/__init__.py
# --------------------------------------------------------------------------------
>>>>>>> origin/main
