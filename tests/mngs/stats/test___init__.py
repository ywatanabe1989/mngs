import pytest
import numpy as np
import pandas as pd
import mngs


class TestStatsModuleImports:
    """Test stats module imports and structure."""

    def test_module_import(self):
        """Test that stats module can be imported."""
        assert hasattr(mngs, "stats")

    def test_submodules(self):
        """Test that submodules are available."""
        assert hasattr(mngs.stats, "desc")
        assert hasattr(mngs.stats, "multiple")
        assert hasattr(mngs.stats, "tests")

    def test_main_functions(self):
        """Test main statistical functions are imported."""
        # Descriptive statistics
        assert hasattr(mngs.stats, "describe")
        assert hasattr(mngs.stats, "nan")
        assert hasattr(mngs.stats, "real")
        
        # Correlation tests
        assert hasattr(mngs.stats, "corr_test")
        assert hasattr(mngs.stats, "corr_test_spearman")
        assert hasattr(mngs.stats, "corr_test_pearson")
        assert hasattr(mngs.stats, "corr_test_multi")
        assert hasattr(mngs.stats, "nocorrelation_test")
        
        # Statistical tests
        assert hasattr(mngs.stats, "brunner_munzel_test")
        assert hasattr(mngs.stats, "smirnov_grubbs")
        
        # P-value formatting
        assert hasattr(mngs.stats, "p2stars")
        
        # Multiple testing corrections
        assert hasattr(mngs.stats, "bonferroni_correction")
        assert hasattr(mngs.stats, "fdr_correction")
        assert hasattr(mngs.stats, "multicompair")

    def test_calc_partial_corr(self):
        """Test partial correlation function is available."""
        assert hasattr(mngs.stats, "calc_partial_corr")

    def test_private_functions_not_exposed(self):
        """Test that private functions are not exposed."""
        # Check that underscore-prefixed functions are not in main namespace
        # (except the ones explicitly imported in __init__.py)
        for attr in dir(mngs.stats):
            if attr.startswith("_") and not attr.startswith("__"):
                # These are explicitly imported in __init__.py
                allowed_private = ["_compute_surrogate"]
                if attr not in allowed_private:
                    # Should not have other private functions
                    assert not callable(getattr(mngs.stats, attr, None))


class TestBasicFunctionality:
    """Test basic functionality of main stats functions."""

    def test_describe_basic(self):
        """Test basic describe functionality."""
        data = np.random.randn(100)
        result = mngs.stats.describe(data)
        
        # Should return a dictionary or similar structure
        assert result is not None
        
        # For pandas Series/DataFrame input
        df = pd.DataFrame({"A": data, "B": data * 2})
        result_df = mngs.stats.describe(df)
        assert result_df is not None

    def test_corr_test_basic(self):
        """Test basic correlation test functionality."""
        x = np.random.randn(50)
        y = 2 * x + np.random.randn(50) * 0.5  # Correlated data
        
        result = mngs.stats.corr_test(x, y)
        assert result is not None
        
        # Test specific correlation types
        result_spear = mngs.stats.corr_test_spearman(x, y)
        assert result_spear is not None
        
        result_pear = mngs.stats.corr_test_pearson(x, y)
        assert result_pear is not None

    def test_p2stars_basic(self):
        """Test p-value to stars conversion."""
        # Test various p-values
        assert mngs.stats.p2stars(0.001) == "***"
        assert mngs.stats.p2stars(0.01) == "**"
        assert mngs.stats.p2stars(0.05) == "*"
        assert mngs.stats.p2stars(0.1) == "n.s."
        
        # Test with array input
        p_values = np.array([0.001, 0.01, 0.05, 0.1, 0.5])
        stars = mngs.stats.p2stars(p_values)
        assert len(stars) == len(p_values)

    def test_multiple_corrections_basic(self):
        """Test multiple testing corrections."""
        p_values = np.array([0.01, 0.04, 0.03, 0.05, 0.20])
        
        # Bonferroni correction
        corrected_bonf = mngs.stats.bonferroni_correction(p_values)
        assert corrected_bonf is not None
        assert len(corrected_bonf) == len(p_values)
        
        # FDR correction
        corrected_fdr = mngs.stats.fdr_correction(p_values)
        assert corrected_fdr is not None

    def test_statistical_tests_basic(self):
        """Test statistical tests with basic data."""
        # Brunner-Munzel test
        x = np.random.randn(30)
        y = np.random.randn(30) + 0.5
        
        result = mngs.stats.brunner_munzel_test(x, y)
        assert result is not None
        
        # Smirnov-Grubbs test for outliers
        data_with_outlier = np.concatenate([np.random.randn(50), [10.0]])
        result = mngs.stats.smirnov_grubbs(data_with_outlier)
        assert result is not None


class TestIntegration:
    """Test integration between stats functions."""

    def test_describe_with_nan_handling(self):
        """Test describe with NaN values."""
        data = np.array([1, 2, 3, np.nan, 5, 6, np.nan])
        
        # Test nan handling
        nan_result = mngs.stats.nan(data)
        assert nan_result is not None
        
        # Test real values only
        real_result = mngs.stats.real(data)
        assert real_result is not None
        assert len(real_result) == 5  # Should have 5 non-nan values

    def test_correlation_workflow(self):
        """Test typical correlation analysis workflow."""
        # Generate correlated data
        n_samples = 100
        x = np.random.randn(n_samples)
        y = 0.7 * x + np.random.randn(n_samples) * 0.5
        z = 0.3 * x + 0.4 * y + np.random.randn(n_samples) * 0.3
        
        # Single correlation test
        corr_xy = mngs.stats.corr_test(x, y)
        assert corr_xy is not None
        
        # Multiple correlation tests
        data = pd.DataFrame({"x": x, "y": y, "z": z})
        corr_multi = mngs.stats.corr_test_multi(data)
        assert corr_multi is not None
        
        # Partial correlation
        partial_corr = mngs.stats.calc_partial_corr(data)
        assert partial_corr is not None

    def test_p_value_workflow(self):
        """Test p-value processing workflow."""
        # Generate multiple p-values
        p_values = np.array([0.001, 0.01, 0.03, 0.04, 0.05, 0.1, 0.5])
        
        # Convert to stars
        stars = mngs.stats.p2stars(p_values)
        assert len(stars) == len(p_values)
        
        # Apply corrections
        p_bonf = mngs.stats.bonferroni_correction(p_values)
        p_fdr = mngs.stats.fdr_correction(p_values)
        
        # Convert corrected p-values to stars
        stars_bonf = mngs.stats.p2stars(p_bonf)
        stars_fdr = mngs.stats.p2stars(p_fdr)
        
        assert len(stars_bonf) == len(p_values)
        assert len(stars_fdr) == len(p_values)


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/stats/__init__.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-17 12:29:22 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/stats/__init__.py
#
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/stats/__init__.py"
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
# from . import desc
# from . import multiple
# from . import tests
# from ._describe_wrapper import describe
# from ._nan_stats import nan, real
# from ._corr_test_wrapper import corr_test, corr_test_spearman, corr_test_pearson
# from .tests._corr_test import _compute_surrogate
# from ._corr_test_multi import corr_test_multi, nocorrelation_test
# from ._statistical_tests import brunner_munzel_test, smirnov_grubbs
# from ._p2stars_wrapper import p2stars
# from ._multiple_corrections import bonferroni_correction, fdr_correction, multicompair
#
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/stats/__init__.py
# --------------------------------------------------------------------------------