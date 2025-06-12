#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-06-11 04:15:00 (ywatanabe)"
# File: ./mngs_repo/tests/mngs/plt/test___init__.py

"""Comprehensive tests for mngs.plt module initialization and matplotlib compatibility."""

import pytest
import sys
import importlib
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock, call
import warnings
import numpy as np
import io
import contextlib


class TestPltModuleImports:
    """Test basic imports and module structure."""
    
    def test_module_imports_successfully(self):
        """Test that mngs.plt module can be imported."""
        import mngs.plt
        assert mngs.plt is not None
        
    def test_submodule_imports(self):
        """Test that submodules are imported correctly."""
        import mngs.plt
        
        # Check required imports
        assert hasattr(mngs.plt, 'subplots')
        assert hasattr(mngs.plt, 'ax')
        assert hasattr(mngs.plt, 'color')
        assert hasattr(mngs.plt, 'close')
        assert hasattr(mngs.plt, 'tpl')
        assert hasattr(mngs.plt, 'enhanced_colorbar')
        
    def test_module_attributes(self):
        """Test module attributes."""
        import mngs.plt
        
        assert hasattr(mngs.plt, '__FILE__')
        assert hasattr(mngs.plt, '__DIR__')
        assert mngs.plt.__FILE__ == "./src/mngs/plt/__init__.py"
        
    def test_matplotlib_import(self):
        """Test that matplotlib is properly imported as _counter_part."""
        import mngs.plt
        
        # Check internal reference to matplotlib
        assert hasattr(mngs.plt, '_counter_part')
        assert mngs.plt._counter_part is plt


class TestMatplotlibCompatibility:
    """Test matplotlib compatibility features."""
    
    def test_getattr_fallback(self):
        """Test __getattr__ fallback to matplotlib.pyplot."""
        import mngs.plt
        
        # Test accessing matplotlib functions through mngs.plt
        assert hasattr(mngs.plt, 'plot')  # matplotlib function
        assert hasattr(mngs.plt, 'scatter')  # matplotlib function
        assert hasattr(mngs.plt, 'xlabel')  # matplotlib function
        
        # These should be the same as matplotlib's
        assert mngs.plt.plot is plt.plot
        assert mngs.plt.scatter is plt.scatter
        assert mngs.plt.xlabel is plt.xlabel
        
    def test_getattr_special_handling(self):
        """Test special handling in __getattr__ for enhanced functions."""
        import mngs.plt
        
        # Test close returns mngs version
        close_func = getattr(mngs.plt, 'close')
        assert close_func is mngs.plt.close
        
        # Test tight_layout returns enhanced version
        tight_layout_func = getattr(mngs.plt, 'tight_layout')
        assert tight_layout_func is mngs.plt.tight_layout
        
        # Test colorbar returns enhanced version
        colorbar_func = getattr(mngs.plt, 'colorbar')
        assert colorbar_func is mngs.plt.enhanced_colorbar
        
    def test_getattr_nonexistent(self):
        """Test __getattr__ raises AttributeError for nonexistent attributes."""
        import mngs.plt
        
        with pytest.raises(AttributeError) as exc_info:
            mngs.plt.nonexistent_function
            
        assert "has attribute 'nonexistent_function'" in str(exc_info.value)
        
    def test_dir_function(self):
        """Test __dir__ returns combined attributes."""
        import mngs.plt
        
        dir_result = dir(mngs.plt)
        
        # Should include local attributes
        assert 'subplots' in dir_result
        assert 'ax' in dir_result
        assert 'color' in dir_result
        assert 'close' in dir_result
        assert 'tpl' in dir_result
        
        # Should include matplotlib attributes
        assert 'plot' in dir_result
        assert 'scatter' in dir_result
        assert 'figure' in dir_result
        assert 'xlabel' in dir_result
        
        # Should be sorted
        assert dir_result == sorted(dir_result)
        
    def test_compatibility_check(self):
        """Test that mngs.plt is compatible with matplotlib.pyplot."""
        import mngs.plt
        
        # Get all matplotlib.pyplot attributes
        pyplot_attrs = set(dir(plt))
        
        # Get all mngs.plt attributes
        mngs_attrs = set(dir(mngs.plt))
        
        # All pyplot attributes should be accessible through mngs.plt
        for attr in pyplot_attrs:
            if not attr.startswith('_'):  # Skip private attributes
                assert hasattr(mngs.plt, attr), f"Missing matplotlib attribute: {attr}"


class TestEnhancedClose:
    """Test enhanced close functionality."""
    
    def test_enhanced_close_patched(self):
        """Test that matplotlib's close is patched."""
        import mngs.plt
        
        # Check that close was patched
        assert plt.close != mngs.plt._original_close
        assert plt.close is mngs.plt._enhanced_close
        
    def test_enhanced_close_no_args(self):
        """Test enhanced close with no arguments."""
        import mngs.plt
        
        with patch.object(mngs.plt, '_original_close') as mock_close:
            plt.close()
            mock_close.assert_called_once_with()
            
    def test_enhanced_close_regular_figure(self):
        """Test enhanced close with regular matplotlib figure."""
        import mngs.plt
        
        fig = MagicMock()
        
        with patch.object(mngs.plt, '_original_close') as mock_close:
            plt.close(fig)
            mock_close.assert_called_once_with(fig)
            
    def test_enhanced_close_figwrapper(self):
        """Test enhanced close with FigWrapper object."""
        import mngs.plt
        
        # Create mock FigWrapper
        fig_wrapper = MagicMock()
        fig_wrapper._fig_mpl = True
        fig_wrapper.figure = MagicMock()
        
        with patch.object(mngs.plt, '_original_close') as mock_close:
            plt.close(fig_wrapper)
            # Should close the underlying figure
            mock_close.assert_called_once_with(fig_wrapper.figure)
            
    def test_enhanced_close_integration(self):
        """Test enhanced close in real scenario."""
        import mngs.plt
        
        # Create a real figure
        fig = plt.figure()
        
        # Close it
        plt.close(fig)
        
        # Figure should be closed
        assert not plt.fignum_exists(fig.number)


class TestEnhancedTightLayout:
    """Test enhanced tight_layout functionality."""
    
    def test_tight_layout_patched(self):
        """Test that matplotlib's tight_layout is patched."""
        import mngs.plt
        
        # Check that tight_layout was patched
        assert plt.tight_layout != mngs.plt._original_tight_layout
        assert plt.tight_layout is mngs.plt.tight_layout
        
    def test_tight_layout_normal_case(self):
        """Test tight_layout in normal case."""
        import mngs.plt
        
        fig = MagicMock()
        fig.get_constrained_layout.return_value = False
        
        with patch('matplotlib.pyplot.gcf', return_value=fig):
            with patch.object(mngs.plt, '_original_tight_layout') as mock_tight:
                plt.tight_layout()
                mock_tight.assert_called_once()
                
    def test_tight_layout_with_constrained_layout(self):
        """Test tight_layout when figure uses constrained_layout."""
        import mngs.plt
        
        fig = MagicMock()
        fig.get_constrained_layout.return_value = True
        
        with patch('matplotlib.pyplot.gcf', return_value=fig):
            with patch.object(mngs.plt, '_original_tight_layout') as mock_tight:
                plt.tight_layout()
                # Should not call original tight_layout
                mock_tight.assert_not_called()
                
    def test_tight_layout_warning_suppression(self):
        """Test that tight_layout suppresses specific warnings."""
        import mngs.plt
        
        fig = MagicMock()
        fig.get_constrained_layout.return_value = False
        
        # Create a warning scenario
        def mock_tight_with_warning(*args, **kwargs):
            warnings.warn("This figure includes Axes that are not compatible with tight_layout")
            
        with patch('matplotlib.pyplot.gcf', return_value=fig):
            with patch.object(mngs.plt, '_original_tight_layout', side_effect=mock_tight_with_warning):
                # Should not raise warning
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    plt.tight_layout()
                    
                    # No warnings should be recorded
                    assert len(w) == 0
                    
    def test_tight_layout_fallback_to_constrained(self):
        """Test fallback to constrained_layout when tight_layout fails."""
        import mngs.plt
        
        fig = MagicMock()
        fig.get_constrained_layout.return_value = False
        
        with patch('matplotlib.pyplot.gcf', return_value=fig):
            with patch.object(mngs.plt, '_original_tight_layout', side_effect=Exception("Layout failed")):
                # Should try to set constrained_layout
                plt.tight_layout()
                
                fig.set_constrained_layout.assert_called_once_with(True)
                fig.set_constrained_layout_pads.assert_called_once()
                
    def test_tight_layout_complete_failure(self):
        """Test when both tight_layout and constrained_layout fail."""
        import mngs.plt
        
        fig = MagicMock()
        fig.get_constrained_layout.return_value = False
        fig.set_constrained_layout.side_effect = Exception("Constrained layout failed")
        
        with patch('matplotlib.pyplot.gcf', return_value=fig):
            with patch.object(mngs.plt, '_original_tight_layout', side_effect=Exception("Layout failed")):
                # Should not raise exception
                plt.tight_layout()
                
                # Both methods should have been tried
                fig.set_constrained_layout.assert_called_once()


class TestModuleOrganization:
    """Test module organization and structure."""
    
    def test_local_module_attributes(self):
        """Test _local_module_attributes is set correctly."""
        import mngs.plt
        
        assert hasattr(mngs.plt, '_local_module_attributes')
        assert isinstance(mngs.plt._local_module_attributes, list)
        
        # Should include our custom attributes
        local_attrs = mngs.plt._local_module_attributes
        assert 'subplots' in local_attrs or 'subplots' in globals()
        
    def test_no_import_side_effects(self):
        """Test that importing doesn't have side effects."""
        # Capture output during import
        f = io.StringIO()
        
        # Remove from cache to force fresh import
        if 'mngs.plt' in sys.modules:
            del sys.modules['mngs.plt']
            
        with contextlib.redirect_stdout(f):
            with contextlib.redirect_stderr(f):
                import mngs.plt
                
        output = f.getvalue()
        
        # Should not print anything
        assert output == ""
        
    def test_submodule_types(self):
        """Test types of imported submodules."""
        import mngs.plt
        
        # ax and color should be modules
        assert hasattr(mngs.plt.ax, '__name__')
        assert hasattr(mngs.plt.color, '__name__')
        
        # Functions should be callable
        assert callable(mngs.plt.subplots)
        assert callable(mngs.plt.close)
        assert callable(mngs.plt.tpl)
        assert callable(mngs.plt.enhanced_colorbar)


class TestIntegrationScenarios:
    """Test real-world integration scenarios."""
    
    def test_basic_plotting_workflow(self):
        """Test basic plotting workflow using mngs.plt."""
        import mngs.plt
        
        # Should work just like matplotlib
        fig, ax = mngs.plt.subplots()
        
        # Plot some data
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)
        
        # Set labels (using matplotlib compatibility)
        mngs.plt.xlabel('X axis')
        mngs.plt.ylabel('Y axis')
        mngs.plt.title('Test Plot')
        
        # Close figure
        mngs.plt.close(fig)
        
    def test_enhanced_features_workflow(self):
        """Test workflow using enhanced features."""
        import mngs.plt
        
        # Create figure
        fig, ax = mngs.plt.subplots()
        
        # Add some content that might cause tight_layout issues
        im = ax.imshow(np.random.rand(10, 10))
        
        # Add colorbar using enhanced version
        cbar = mngs.plt.colorbar(im)
        
        # Use enhanced tight_layout (should handle colorbar)
        mngs.plt.tight_layout()
        
        # Close using enhanced close
        mngs.plt.close(fig)
        
    def test_module_reload(self):
        """Test that module can be reloaded."""
        import mngs.plt
        
        # Store reference to a function
        original_subplots = mngs.plt.subplots
        
        # Reload module
        importlib.reload(mngs.plt)
        
        # Should still have all functions
        assert hasattr(mngs.plt, 'subplots')
        assert hasattr(mngs.plt, 'close')
        assert hasattr(mngs.plt, 'tight_layout')
        
        # Matplotlib compatibility should still work
        assert hasattr(mngs.plt, 'plot')
        assert hasattr(mngs.plt, 'scatter')


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_attribute_error_message(self):
        """Test that AttributeError has helpful message."""
        import mngs.plt
        
        with pytest.raises(AttributeError) as exc_info:
            mngs.plt.completely_fake_function
            
        error_msg = str(exc_info.value)
        assert "module 'mngs.plt'" in error_msg
        assert "matplotlib.pyplot" in error_msg
        assert "completely_fake_function" in error_msg
        
    def test_no_attribute_leakage(self):
        """Test that private attributes aren't exposed."""
        import mngs.plt
        
        # Private attributes should not be accessible through __getattr__
        with pytest.raises(AttributeError):
            mngs.plt._some_private_attr
            
    def test_import_error_handling(self):
        """Test handling of import errors."""
        # This tests the robustness of the module structure
        import mngs.plt
        
        # Even if submodules have issues, main module should work
        assert mngs.plt is not None
        assert callable(getattr(mngs.plt, '__getattr__', None))
        assert callable(getattr(mngs.plt, '__dir__', None))


class TestPerformance:
    """Test performance-related aspects."""
    
    def test_import_time(self):
        """Test that module imports quickly."""
        import time
        
        # Remove from cache
        if 'mngs.plt' in sys.modules:
            del sys.modules['mngs.plt']
            
        start = time.time()
        import mngs.plt
        end = time.time()
        
        # Should import reasonably fast
        assert (end - start) < 2.0  # 2 seconds is generous
        
    def test_attribute_access_performance(self):
        """Test that attribute access is efficient."""
        import mngs.plt
        import time
        
        # Access matplotlib function many times
        start = time.time()
        for _ in range(1000):
            _ = mngs.plt.plot
        end = time.time()
        
        # Should be fast (caching should help)
        assert (end - start) < 0.1  # 100ms for 1000 accesses
        
    def test_dir_performance(self):
        """Test that dir() is reasonably fast."""
        import mngs.plt
        import time
        
        start = time.time()
        for _ in range(100):
            _ = dir(mngs.plt)
        end = time.time()
        
        # Should be reasonably fast
        assert (end - start) < 1.0  # 1 second for 100 calls


if __name__ == "__main__":
<<<<<<< HEAD
    pytest.main([__file__, "-v"])
=======
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/plt/__init__.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-03 11:53:18 (ywatanabe)"
# # File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/__init__.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/plt/__init__.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# from ._subplots._SubplotsWrapper import subplots
# from . import ax
# from . import color
# from .utils._close import close
# from ._tpl import tpl
# 
# 
# ################################################################################
# # For Matplotlib Compatibility
# ################################################################################
# import matplotlib.pyplot as _counter_part
# 
# _local_module_attributes = list(globals().keys())
# 
# 
# def __getattr__(name):
#     """
#     Fallback to fetch attributes from matplotlib.pyplot
#     if they are not defined directly in this module.
#     """
#     try:
#         # Get the attribute from matplotlib.pyplot
#         return getattr(_counter_part, name)
#     except AttributeError:
#         # Raise the standard error if not found in pyplot either
#         raise AttributeError(
#             f"module '{__name__}' nor matplotlib.pyplot has attribute '{name}'"
#         ) from None
# 
# 
# def __dir__():
#     """
#     Provide combined directory for tab completion, including
#     attributes from this module and matplotlib.pyplot.
#     """
#     # Get attributes defined explicitly in this module
#     local_attrs = set(_local_module_attributes)
#     # Get attributes from matplotlib.pyplot
#     pyplot_attrs = set(dir(_counter_part))
#     # Return the sorted union
#     return sorted(local_attrs.union(pyplot_attrs))
# 
# 
# """
# import matplotlib.pyplot as _counter_part
# import mngs.plt as mplt
# 
# set(dir(mplt)) - set(dir(_counter_part))
# set(dir(_counter_part))
# 
# mplt.yticks
# 
# is_compatible = np.all([kk in set(dir(mplt)) for kk in set(dir(_counter_part))])
# if is_compatible:
#     print(f"{mplt.__name__} is compatible with {_counter_part.__name__}")
# else:
#     print(f"{mplt.__name__} is incompatible with {_counter_part.__name__}")
# """
# 
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/plt/__init__.py
# --------------------------------------------------------------------------------
>>>>>>> origin/main
