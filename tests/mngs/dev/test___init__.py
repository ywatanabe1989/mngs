#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-06-02 16:20:00 (ywatanabe)"
# File: tests/mngs/dev/test___init__.py

import pytest
from unittest.mock import patch, MagicMock, mock_open
import threading
import time
import sys


class TestDevModule:
    """Test suite for mngs.dev module."""

    def test_code_flow_analyzer_import(self):
        """Test that CodeFlowAnalyzer can be imported from mngs.dev."""
        from mngs.dev import CodeFlowAnalyzer
        
        assert CodeFlowAnalyzer is not None
        assert hasattr(CodeFlowAnalyzer, '__init__')

    def test_reload_import(self):
        """Test that reload function can be imported from mngs.dev."""
        from mngs.dev import reload
        
        assert callable(reload)
        assert hasattr(reload, '__call__')

    def test_reload_auto_import(self):
        """Test that reload_auto function can be imported from mngs.dev."""
        from mngs.dev import reload_auto
        
        assert callable(reload_auto)
        assert hasattr(reload_auto, '__call__')

    def test_module_attributes(self):
        """Test that mngs.dev module has expected attributes."""
        import mngs.dev
        
        assert hasattr(mngs.dev, 'CodeFlowAnalyzer')
        assert hasattr(mngs.dev, 'reload')
        assert hasattr(mngs.dev, 'reload_auto')
        
        # Check that they're the right types
        assert isinstance(mngs.dev.CodeFlowAnalyzer, type)  # It's a class
        assert callable(mngs.dev.reload)
        assert callable(mngs.dev.reload_auto)

    def test_dynamic_import_mechanism(self):
        """Test that the dynamic import mechanism works correctly."""
        import mngs.dev
        
        # Check that functions/classes are available after dynamic import
        assert hasattr(mngs.dev, 'CodeFlowAnalyzer')
        assert hasattr(mngs.dev, 'reload')
        assert hasattr(mngs.dev, 'reload_auto')
        
        # Check that cleanup variables are not present
        assert not hasattr(mngs.dev, 'os')
        assert not hasattr(mngs.dev, 'importlib')
        assert not hasattr(mngs.dev, 'inspect')
        assert not hasattr(mngs.dev, 'current_dir')

    def test_code_flow_analyzer_initialization(self):
        """Test CodeFlowAnalyzer class initialization."""
        from mngs.dev import CodeFlowAnalyzer
        
        test_file_path = "/path/to/test/file.py"
        analyzer = CodeFlowAnalyzer(test_file_path)
        
        assert analyzer.file_path == test_file_path
        assert hasattr(analyzer, 'execution_flow')
        assert hasattr(analyzer, 'sequence')
        assert hasattr(analyzer, 'skip_functions')
        
        # Check initial values
        assert analyzer.execution_flow == []
        assert analyzer.sequence == 1
        assert isinstance(analyzer.skip_functions, (set, list, dict))

    def test_code_flow_analyzer_skip_functions(self):
        """Test that CodeFlowAnalyzer has expected skip functions."""
        from mngs.dev import CodeFlowAnalyzer
        
        analyzer = CodeFlowAnalyzer("test.py")
        
        # Check that common built-in functions are in skip list
        skip_functions = analyzer.skip_functions
        expected_skips = ["__init__", "__main__", "print", "len", "str"]
        
        for func in expected_skips:
            assert func in skip_functions, f"Function '{func}' should be in skip_functions"

    def test_reload_basic_functionality(self):
        """Test basic reload functionality with mocked modules."""
        from mngs.dev import reload
        
        # Mock sys.modules to simulate mngs modules
        original_modules = sys.modules.copy()
        
        try:
            # Add fake mngs modules to sys.modules
            fake_mngs = MagicMock()
            fake_mngs_sub = MagicMock()
            sys.modules['mngs'] = fake_mngs
            sys.modules['mngs.fake_submodule'] = fake_mngs_sub
            
            with patch('importlib.reload') as mock_reload:
                # Mock the final mngs reload to return the fake module
                mock_reload.return_value = fake_mngs
                
                result = reload()
                
                # Should have called reload on mngs modules
                assert mock_reload.called
                assert result is not None
                
        finally:
            # Restore original sys.modules
            sys.modules.clear()
            sys.modules.update(original_modules)

    def test_reload_handles_exceptions(self):
        """Test that reload handles exceptions gracefully."""
        from mngs.dev import reload
        
        original_modules = sys.modules.copy()
        
        try:
            # Add fake mngs modules that will raise exceptions
            fake_problematic_module = MagicMock()
            sys.modules['mngs'] = MagicMock()
            sys.modules['mngs.problematic'] = fake_problematic_module
            
            with patch('importlib.reload') as mock_reload:
                # Make some reloads fail
                def side_effect(module):
                    if module == fake_problematic_module:
                        raise ImportError("Test error")
                    return module
                
                mock_reload.side_effect = side_effect
                
                # Should not raise exception despite individual failures
                result = reload()
                assert result is not None
                
        finally:
            sys.modules.clear()
            sys.modules.update(original_modules)

    def test_reload_auto_basic_functionality(self):
        """Test reload_auto basic functionality with mocking."""
        from mngs.dev import reload_auto
        
        with patch('mngs.dev._reload.reload') as mock_reload:
            with patch('threading.Thread') as mock_thread:
                with patch('time.sleep') as mock_sleep:
                    
                    # Create a mock thread
                    mock_thread_instance = MagicMock()
                    mock_thread.return_value = mock_thread_instance
                    
                    # Call reload_auto with short interval
                    reload_auto(interval=1)
                    
                    # Should create a thread
                    mock_thread.assert_called_once()
                    
                    # Should start the thread
                    mock_thread_instance.start.assert_called_once()

    def test_reload_auto_custom_interval(self):
        """Test reload_auto with custom interval."""
        from mngs.dev import reload_auto
        
        with patch('threading.Thread') as mock_thread:
            mock_thread_instance = MagicMock()
            mock_thread.return_value = mock_thread_instance
            
            # Test with custom interval
            custom_interval = 5
            reload_auto(interval=custom_interval)
            
            # Check that thread was created
            mock_thread.assert_called_once()
            mock_thread_instance.start.assert_called_once()

    def test_reload_thread_management(self):
        """Test that reload_auto properly manages threads."""
        from mngs.dev import reload_auto
        import mngs.dev._reload as reload_module
        
        # Reset module state
        reload_module._running = False
        reload_module._reload_thread = None
        
        with patch('threading.Thread') as mock_thread:
            with patch('time.sleep') as mock_sleep:
                mock_thread_instance = MagicMock()
                mock_thread.return_value = mock_thread_instance
                
                # Start auto-reload
                reload_auto(interval=1)
                
                # Should create and start thread
                assert mock_thread.called
                assert mock_thread_instance.start.called

    def test_code_flow_analyzer_methods(self):
        """Test that CodeFlowAnalyzer has expected methods."""
        from mngs.dev import CodeFlowAnalyzer
        
        analyzer = CodeFlowAnalyzer("test.py")
        
        # Check for expected methods (these might vary based on implementation)
        methods = dir(analyzer)
        
        # Should have basic Python object methods
        assert '__init__' in methods
        assert '__dict__' in methods or hasattr(analyzer, '__dict__')

    def test_reload_function_signature(self):
        """Test reload function signature."""
        from mngs.dev import reload
        import inspect
        
        sig = inspect.signature(reload)
        params = list(sig.parameters.keys())
        
        # reload() should take no parameters
        assert len(params) == 0

    def test_reload_auto_function_signature(self):
        """Test reload_auto function signature."""
        from mngs.dev import reload_auto
        import inspect
        
        sig = inspect.signature(reload_auto)
        params = list(sig.parameters.keys())
        
        # Should have interval parameter
        assert 'interval' in params
        
        # Check default value
        interval_param = sig.parameters['interval']
        assert interval_param.default == 10

    def test_code_flow_analyzer_with_mock_file(self):
        """Test CodeFlowAnalyzer with mocked file system."""
        from mngs.dev import CodeFlowAnalyzer
        
        fake_file_path = "/fake/path/test.py"
        
        # Should not crash even with non-existent file
        analyzer = CodeFlowAnalyzer(fake_file_path)
        assert analyzer.file_path == fake_file_path

    def test_reload_return_type(self):
        """Test that reload returns something (module-like object)."""
        from mngs.dev import reload
        
        with patch('importlib.reload') as mock_reload:
            # Mock to return a fake module
            fake_mngs = MagicMock()
            mock_reload.return_value = fake_mngs
            
            result = reload()
            
            # Should return the reloaded module
            assert result is not None

    def test_module_docstrings(self):
        """Test that imported classes/functions have docstrings."""
        from mngs.dev import CodeFlowAnalyzer, reload, reload_auto
        
        # Check docstrings exist
        assert hasattr(reload, '__doc__')
        assert reload.__doc__ is not None
        assert 'reload' in reload.__doc__.lower()

    def test_dev_module_integration(self):
        """Test integration between dev module components."""
        from mngs.dev import CodeFlowAnalyzer, reload, reload_auto
        
        # All should be importable and callable/instantiable
        analyzer = CodeFlowAnalyzer("test.py")
        assert analyzer is not None
        
        # Functions should be callable
        assert callable(reload)
        assert callable(reload_auto)

    def test_threading_safety(self):
        """Test that reload functions handle threading safely."""
        from mngs.dev import reload_auto
        import mngs.dev._reload as reload_module
        
        # Reset state
        reload_module._running = False
        reload_module._reload_thread = None
        
        with patch('threading.Thread') as mock_thread:
            mock_thread_instance = MagicMock()
            mock_thread.return_value = mock_thread_instance
            
            # Should be able to call multiple times safely
            reload_auto(interval=1)
            reload_auto(interval=2)
            
            # Should create threads
            assert mock_thread.called

    def test_code_flow_analyzer_file_path_handling(self):
        """Test CodeFlowAnalyzer handles different file path formats."""
        from mngs.dev import CodeFlowAnalyzer
        
        test_paths = [
            "/absolute/path/file.py",
            "relative/path/file.py",
            "file.py",
            "/path/with spaces/file.py",
            "/path/with-dashes/file_with_underscores.py"
        ]
        
        for path in test_paths:
            analyzer = CodeFlowAnalyzer(path)
            assert analyzer.file_path == path

    def test_reload_module_filtering(self):
        """Test that reload only affects mngs modules."""
        from mngs.dev import reload
        
        original_modules = sys.modules.copy()
        
        try:
            # Add both mngs and non-mngs modules
            sys.modules['mngs'] = MagicMock()
            sys.modules['mngs.test'] = MagicMock()
            sys.modules['numpy'] = MagicMock()  # Non-mngs module
            sys.modules['other_package'] = MagicMock()  # Non-mngs module
            
            with patch('importlib.reload') as mock_reload:
                mock_reload.return_value = MagicMock()
                
                reload()
                
                # Should only reload mngs modules
                reloaded_modules = [call[0][0] for call in mock_reload.call_args_list]
                
                # Check that only mngs modules were reloaded
                for module in reloaded_modules:
                    if hasattr(module, '__name__'):
                        module_name = module.__name__
                    else:
                        # For mock objects, we need to check differently
                        continue
                    
                    # Skip the final mngs reload
                    if module_name and not module_name.startswith('mngs'):
                        pytest.fail(f"Non-mngs module {module_name} was reloaded")
                        
        finally:
            sys.modules.clear()
            sys.modules.update(original_modules)


if __name__ == "__main__":
    import os
<<<<<<< HEAD
    pytest.main([os.path.abspath(__file__)])
=======

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/dev/__init__.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-07 17:06:37 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/dev/__init__.py
# 
# 
# import importlib
# import inspect
# import os
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
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/dev/__init__.py
# --------------------------------------------------------------------------------
>>>>>>> origin/main
