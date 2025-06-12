#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-06-02 15:27:00 (ywatanabe)"
# File: ./tests/mngs/torch/test___init__.py

"""
Functionality:
    * Tests torch module initialization and function imports
    * Validates PyTorch utility function availability
    * Tests module structure and import consistency
Input:
    * Module imports
Output:
    * Test results
Prerequisites:
    * pytest, torch
"""

import pytest
import inspect


class TestTorchModuleInit:
    """Test cases for torch module initialization."""

    def setup_method(self):
        """Setup test fixtures."""
        # Skip tests if torch not available
        pytest.importorskip("torch")

    def test_module_imports_successfully(self):
        """Test that the torch module imports without errors."""
        import mngs.torch
        assert mngs.torch is not None

    def test_apply_to_function_available(self):
        """Test that apply_to function is available."""
        import mngs.torch
        assert hasattr(mngs.torch, 'apply_to')
        assert callable(mngs.torch.apply_to)

    def test_nan_functions_available(self):
        """Test that NaN-handling functions are available."""
        import mngs.torch
        nan_functions = [
            'nanmax', 'nanmin', 'nanvar', 'nanstd', 
            'nanprod', 'nancumprod', 'nancumsum', 
            'nanargmin', 'nanargmax'
        ]
        
        for func_name in nan_functions:
            assert hasattr(mngs.torch, func_name), f"Function {func_name} not found"
            assert callable(getattr(mngs.torch, func_name)), f"{func_name} is not callable"

    def test_function_signatures(self):
        """Test that functions have expected signatures."""
        import mngs.torch
        
        # Test apply_to signature
        sig = inspect.signature(mngs.torch.apply_to)
        params = list(sig.parameters.keys())
        assert 'fn' in params
        assert 'x' in params
        assert 'dim' in params

        # Test nanmax signature
        sig = inspect.signature(mngs.torch.nanmax)
        params = list(sig.parameters.keys())
        assert 'tensor' in params
        assert 'dim' in params
        assert 'keepdim' in params

    def test_module_has_proper_imports(self):
        """Test that module imports the expected submodules."""
        import mngs.torch
        
        # Check that functions come from proper modules
        assert mngs.torch.apply_to.__module__ == 'mngs.torch._apply_to'
        assert mngs.torch.nanmax.__module__ == 'mngs.torch._nan_funcs'

    def test_no_unwanted_attributes(self):
        """Test that module doesn't expose unwanted attributes."""
        import mngs.torch
        
        # Get all attributes
        attrs = dir(mngs.torch)
        
        # Should not have these internal attributes exposed
        unwanted = ['torch', '_torch', 'warnings']
        for attr in unwanted:
            assert attr not in attrs, f"Unwanted attribute {attr} exposed"

    def test_module_reimport_consistency(self):
        """Test that multiple imports are consistent."""
        import mngs.torch as torch1
        import mngs.torch as torch2
        
        assert torch1 is torch2
        assert torch1.apply_to is torch2.apply_to
        assert torch1.nanmax is torch2.nanmax

    def test_torch_dependency_handling(self):
        """Test graceful handling of torch dependency."""
        # This test ensures the module can handle torch availability
        try:
            import torch
            import mngs.torch
            # If torch is available, functions should work
            assert hasattr(mngs.torch, 'apply_to')
        except ImportError:
            # If torch not available, should still be importable
            # (though functions may not work)
            import mngs.torch
            assert mngs.torch is not None

    def test_star_import_from_nan_funcs(self):
        """Test that star import from _nan_funcs works correctly."""
        import mngs.torch
        
        # All nan functions should be available at module level
        expected_functions = [
            'nanmax', 'nanmin', 'nanvar', 'nanstd',
            'nanprod', 'nancumprod', 'nancumsum', 
            'nanargmin', 'nanargmax'
        ]
        
        for func in expected_functions:
            assert hasattr(mngs.torch, func)
            # Should be the same object as in _nan_funcs
            from mngs.torch._nan_funcs import nanmax
            if func == 'nanmax':
                assert getattr(mngs.torch, func) is nanmax


if __name__ == "__main__":
    import os
    import pytest
    pytest.main([os.path.abspath(__file__)])