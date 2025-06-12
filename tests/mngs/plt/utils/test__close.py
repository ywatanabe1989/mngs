#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-06-02 15:36:00 (ywatanabe)"
# File: ./mngs_repo/tests/mngs/plt/utils/test__close.py

"""Tests for close functionality."""

import pytest
import matplotlib
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock, Mock

from mngs.plt.utils._close import close


class TestClose:
    """Test close function."""

    @patch('mngs.plt.utils._close.plt.close')
    def test_close_matplotlib_figure(self, mock_plt_close):
        """Test closing a standard matplotlib Figure object."""
        # Create a mock matplotlib figure
        mock_figure = Mock(spec=matplotlib.figure.Figure)
        
        # Call close function
        close(mock_figure)
        
        # Verify plt.close was called with the figure
        mock_plt_close.assert_called_once_with(mock_figure)
        
    @patch('mngs.plt.utils._close.plt.close')
    def test_close_mngs_figwrapper(self, mock_plt_close):
        """Test closing an MNGS FigWrapper object."""
        # Create a mock FigWrapper with a figure attribute
        mock_figwrapper = MagicMock()
        mock_figwrapper.figure = Mock(spec=matplotlib.figure.Figure)
        
        # Mock the isinstance check for FigWrapper
        with patch('mngs.plt.utils._close.isinstance') as mock_isinstance:
            def isinstance_side_effect(obj, cls):
                if obj is mock_figwrapper and 'FigWrapper' in str(cls):
                    return True
                if obj is mock_figwrapper and cls is matplotlib.figure.Figure:
                    return False
                return isinstance(obj, cls)
            mock_isinstance.side_effect = isinstance_side_effect
            
            # Call close function
            close(mock_figwrapper)
            
            # Verify plt.close was called with the underlying figure
            mock_plt_close.assert_called_once_with(mock_figwrapper.figure)
            
    def test_close_invalid_object_type_error(self):
        """Test that TypeError is raised for invalid object types."""
        invalid_objects = [
            "string",
            123,
            [],
            {},
            None,
            object(),
            MagicMock()  # Generic mock without proper spec
        ]
        
        for invalid_obj in invalid_objects:
            with pytest.raises(TypeError) as exc_info:
                close(invalid_obj)
            
            # Check error message content
            error_msg = str(exc_info.value)
            assert "Cannot close object of type" in error_msg
            assert "Expected FigWrapper or Figure object" in error_msg
            assert type(invalid_obj).__name__ in error_msg
            
    @patch('mngs.plt.utils._close.plt.close')
    def test_close_real_matplotlib_figure(self, mock_plt_close):
        """Test closing with a real matplotlib figure (using mock to avoid GUI)."""
        # Use matplotlib's Figure class directly
        from matplotlib.figure import Figure
        
        # Create actual figure instance (without GUI backend)
        fig = Figure(figsize=(6, 4))
        
        # Call close function
        close(fig)
        
        # Verify plt.close was called
        mock_plt_close.assert_called_once_with(fig)
        
    def test_close_type_checking_specificity(self):
        """Test that type checking is specific and correct."""
        # Test with object that inherits from Figure
        class CustomFigure(matplotlib.figure.Figure):
            def __init__(self):
                # Don't call super().__init__ to avoid matplotlib setup
                pass
                
        with patch('mngs.plt.utils._close.plt.close') as mock_plt_close:
            custom_fig = CustomFigure()
            close(custom_fig)
            mock_plt_close.assert_called_once_with(custom_fig)
            
    def test_close_error_message_formatting(self):
        """Test error message formatting for different object types."""
        test_objects = [
            ("string", str),
            (42, int),
            ([1, 2, 3], list),
            ({"key": "value"}, dict)
        ]
        
        for obj, expected_type in test_objects:
            with pytest.raises(TypeError) as exc_info:
                close(obj)
            
            error_msg = str(exc_info.value)
            assert f"object of type {expected_type.__name__}" in error_msg
            
    @patch('mngs.plt.utils._close.plt.close')
    def test_close_memory_management_pattern(self, mock_plt_close):
        """Test close function in typical memory management pattern."""
        # Simulate creating multiple figures and closing them
        figures = []
        
        for i in range(5):
            mock_fig = Mock(spec=matplotlib.figure.Figure)
            figures.append(mock_fig)
            close(mock_fig)
            
        # Verify all figures were closed
        assert mock_plt_close.call_count == 5
        for i, fig in enumerate(figures):
            assert mock_plt_close.call_args_list[i][0][0] is fig
            
    def test_close_function_signature(self):
        """Test that close function has the expected signature."""
        import inspect
        
        sig = inspect.signature(close)
        params = list(sig.parameters.keys())
        
        # Should have exactly one parameter named 'obj'
        assert len(params) == 1
        assert params[0] == 'obj'
        
        # Parameter should not have a default value
        assert sig.parameters['obj'].default == inspect.Parameter.empty
        
    @patch('mngs.plt.utils._close.plt.close')
    def test_close_exception_propagation(self, mock_plt_close):
        """Test that exceptions from plt.close are propagated."""
        # Make plt.close raise an exception
        mock_plt_close.side_effect = RuntimeError("Mock error from plt.close")
        
        mock_figure = Mock(spec=matplotlib.figure.Figure)
        
        # close should propagate the exception
        with pytest.raises(RuntimeError, match="Mock error from plt.close"):
            close(mock_figure)
            
    def test_close_imports_availability(self):
        """Test that required imports are available."""
        # Test that the function can access required modules
        import mngs.plt.utils._close as close_module
        
        assert hasattr(close_module, 'matplotlib')
        assert hasattr(close_module, 'plt')
        assert hasattr(close_module, 'mngs_plt')
        
    @patch('mngs.plt.utils._close.plt.close')
    def test_close_with_none_figure_attribute(self, mock_plt_close):
        """Test handling when FigWrapper has None figure attribute."""
        # Create mock FigWrapper with None figure
        mock_figwrapper = MagicMock()
        mock_figwrapper.figure = None
        
        with patch('mngs.plt.utils._close.isinstance') as mock_isinstance:
            def isinstance_side_effect(obj, cls):
                if obj is mock_figwrapper and 'FigWrapper' in str(cls):
                    return True
                if obj is mock_figwrapper and cls is matplotlib.figure.Figure:
                    return False
                return isinstance(obj, cls)
            mock_isinstance.side_effect = isinstance_side_effect
            
            # Should still call plt.close even with None
            close(mock_figwrapper)
            mock_plt_close.assert_called_once_with(None)
            
    def test_close_docstring_content(self):
        """Test that close function has comprehensive docstring."""
        assert close.__doc__ is not None
        doc = close.__doc__
        
        # Check for key documentation elements
        assert "Close a matplotlib figure" in doc
        assert "Parameters" in doc
        assert "Raises" in doc
        assert "Examples" in doc
        assert "TypeError" in doc
        assert "memory leaks" in doc.lower()
        
    @patch('mngs.plt.utils._close.plt.close')
    def test_close_integration_with_mock_figwrapper_class(self, mock_plt_close):
        """Test close with properly mocked FigWrapper class."""
        # Mock the mngs_plt module and FigWrapper class
        with patch('mngs.plt.utils._close.mngs_plt') as mock_mngs_plt:
            # Create a mock FigWrapper class
            mock_figwrapper_class = MagicMock()
            mock_mngs_plt._subplots._FigWrapper.FigWrapper = mock_figwrapper_class
            
            # Create instance that will be recognized as FigWrapper
            mock_figwrapper_instance = MagicMock()
            mock_figwrapper_instance.figure = Mock(spec=matplotlib.figure.Figure)
            
            # Make isinstance return True for our mock
            with patch('mngs.plt.utils._close.isinstance') as mock_isinstance:
                def isinstance_side_effect(obj, cls):
                    if obj is mock_figwrapper_instance and cls is mock_figwrapper_class:
                        return True
                    if obj is mock_figwrapper_instance and cls is matplotlib.figure.Figure:
                        return False
                    return False
                mock_isinstance.side_effect = isinstance_side_effect
                
                close(mock_figwrapper_instance)
                mock_plt_close.assert_called_once_with(mock_figwrapper_instance.figure)


if __name__ == "__main__":
    import os
<<<<<<< HEAD
    pytest.main([os.path.abspath(__file__)])
=======

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/plt/utils/_close.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-04-29 20:41:30 (ywatanabe)"
# # File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/_close.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/plt/_close.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# import matplotlib
# import matplotlib.pyplot as plt
# import mngs.plt as mngs_plt
# 
# 
# def close(obj):
#     if isinstance(obj, matplotlib.figure.Figure):
#         plt.close(obj)
#     elif isinstance(obj, mngs_plt._subplots._FigWrapper.FigWrapper):
#         plt.close(obj.figure)
#     else:
#         raise TypeError(
#             f"Cannot close object of type {type(obj).__name__}. Expected FigWrapper or Figure object."
#         )
# 
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/plt/utils/_close.py
# --------------------------------------------------------------------------------
>>>>>>> origin/main
