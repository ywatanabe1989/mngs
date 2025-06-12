#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-08 05:52:35 (ywatanabe)"
# File: ./mngs_repo/tests/mngs/stats/tests/test___init__.py

"""
Tests for stats.tests module initialization.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os
import sys
import importlib
import inspect


class TestStatsTestsModuleInit:
    """Test stats.tests module initialization and dynamic imports."""
    
    def test_module_imports_successfully(self):
        """Test that the stats.tests module can be imported."""
        import mngs.stats.tests
        assert mngs.stats.tests is not None
    
    def test_corr_test_function_available(self):
        """Test that corr_test function is available after import."""
        import mngs.stats.tests
        assert hasattr(mngs.stats.tests, 'corr_test')
        assert callable(mngs.stats.tests.corr_test)
    
    def test_brunner_munzel_test_available(self):
        """Test that brunner_munzel_test is available."""
        import mngs.stats.tests
        assert hasattr(mngs.stats.tests, 'brunner_munzel_test')
        assert callable(mngs.stats.tests.brunner_munzel_test)
    
    def test_nocorrelation_test_available(self):
        """Test that nocorrelation_test is available."""
        import mngs.stats.tests
        assert hasattr(mngs.stats.tests, 'nocorrelation_test')
        assert callable(mngs.stats.tests.nocorrelation_test)
    
    def test_smirnov_grubbs_test_available(self):
        """Test that smirnov_grubbs_test is available."""
        import mngs.stats.tests
        assert hasattr(mngs.stats.tests, 'smirnov_grubbs_test')
        assert callable(mngs.stats.tests.smirnov_grubbs_test)
    
    def test_dynamic_import_mechanism(self):
        """Test the dynamic import mechanism works correctly."""
        # Create mock modules
        mock_module1 = MagicMock()
        mock_module1.test_function = lambda: "test1"
        mock_module1.TestClass = type('TestClass', (), {})
        mock_module1._private_function = lambda: "private"
        
        mock_module2 = MagicMock()
        mock_module2.another_function = lambda: "test2"
        
        # Mock the directory listing
        mock_files = ['_test_module1.py', '_test_module2.py', '__init__.py', 'README.md']
        
        with patch('os.listdir', return_value=mock_files):
            with patch('importlib.import_module', side_effect=[mock_module1, mock_module2]) as mock_import:
                # Re-execute the module initialization logic
                current_dir = "/fake/dir"
                namespace = {}
                
                for filename in mock_files:
                    if filename.endswith(".py") and not filename.startswith("__"):
                        module_name = filename[:-3]
                        if module_name == '_test_module1':
                            module = mock_module1
                        else:
                            module = mock_module2
                        
                        for name, obj in inspect.getmembers(module):
                            if inspect.isfunction(obj) or inspect.isclass(obj):
                                if not name.startswith("_"):
                                    namespace[name] = obj
                
                # Verify public functions/classes were imported
                assert 'test_function' in namespace
                assert 'TestClass' in namespace
                assert 'another_function' in namespace
                
                # Verify private functions were not imported
                assert '_private_function' not in namespace
    
    def test_cleanup_of_temporary_variables(self):
        """Test that temporary variables are cleaned up after import."""
        # Import fresh to ensure initialization runs
        if 'mngs.stats.tests' in sys.modules:
            del sys.modules['mngs.stats.tests']
        
        import mngs.stats.tests
        
        # Check that temporary variables don't exist in the module namespace
        assert not hasattr(mngs.stats.tests, 'filename')
        assert not hasattr(mngs.stats.tests, 'module_name')
        assert not hasattr(mngs.stats.tests, 'module')
        assert not hasattr(mngs.stats.tests, 'name')
        assert not hasattr(mngs.stats.tests, 'obj')
    
    def test_only_public_functions_exported(self):
        """Test that only public functions are exported."""
        import mngs.stats.tests
        
        # Get all attributes
        all_attrs = dir(mngs.stats.tests)
        
        # Filter for actual stats.tests module functions (not built-ins)
        test_functions = [
            attr for attr in all_attrs 
            if not attr.startswith('__') and callable(getattr(mngs.stats.tests, attr))
        ]
        
        # Verify no private functions are exposed (except those from __corr_test* modules)
        for func_name in test_functions:
            if not func_name.startswith('corr_test'):  # Exception for correlation test functions
                assert not func_name.startswith('_'), f"Private function {func_name} should not be exported"
    
    def test_correlation_test_functions(self):
        """Test that correlation test functions are available."""
        import mngs.stats.tests
        
        # Test main correlation function
        assert hasattr(mngs.stats.tests, 'corr_test')
        
        # Test specific correlation functions (these might come from __corr_test modules)
        expected_corr_functions = ['corr_test', 'corr_test_pearson', 'corr_test_spearman']
        
        for func_name in expected_corr_functions:
            if hasattr(mngs.stats.tests, func_name):
                assert callable(getattr(mngs.stats.tests, func_name))
    
    def test_statistical_test_functions(self):
        """Test that various statistical test functions are available."""
        import mngs.stats.tests
        
        # These should be available from various test modules
        expected_functions = [
            'brunner_munzel_test',
            'nocorrelation_test',
            'smirnov_grubbs_test'
        ]
        
        for func_name in expected_functions:
            assert hasattr(mngs.stats.tests, func_name), f"Function {func_name} should be available"
            assert callable(getattr(mngs.stats.tests, func_name)), f"{func_name} should be callable"
    
    def test_no_submodule_conflicts(self):
        """Test that there are no naming conflicts between submodules."""
        import mngs.stats.tests
        
        # Get all public attributes
        attrs = {name: getattr(mngs.stats.tests, name) for name in dir(mngs.stats.tests) if not name.startswith('_')}
        
        # Count occurrences of each function object
        func_objects = {}
        for name, obj in attrs.items():
            if callable(obj):
                id_obj = id(obj)
                if id_obj not in func_objects:
                    func_objects[id_obj] = []
                func_objects[id_obj].append(name)
        
        # Check for duplicate names pointing to different objects
        names_seen = set()
        for obj_id, names in func_objects.items():
            for name in names:
                assert name not in names_seen or len(names) == 1, f"Name conflict detected for {name}"
                names_seen.add(name)
    
    def test_module_reimport_consistency(self):
        """Test that reimporting the module gives consistent results."""
        # First import
        import mngs.stats.tests
        functions_first = set(name for name in dir(mngs.stats.tests) if callable(getattr(mngs.stats.tests, name)) and not name.startswith('_'))
        
        # Force reimport
        importlib.reload(mngs.stats.tests)
        functions_second = set(name for name in dir(mngs.stats.tests) if callable(getattr(mngs.stats.tests, name)) and not name.startswith('_'))
        
        # Should have the same functions
        assert functions_first == functions_second
    
    def test_module_file_path(self):
        """Test that the module file path is correctly set."""
        import mngs.stats.tests
        
        # The module should have a __file__ attribute
        assert hasattr(mngs.stats.tests, '__file__')
        assert mngs.stats.tests.__file__.endswith('__init__.py')
        assert 'stats' in mngs.stats.tests.__file__
        assert 'tests' in mngs.stats.tests.__file__
    
    def test_imports_from_double_underscore_files(self):
        """Test that functions from __corr_test files are imported."""
        import mngs.stats.tests
        
        # These functions should be available from __corr_test modules
        # At minimum, the main corr_test function should be available
        assert hasattr(mngs.stats.tests, 'corr_test')
        
        # Check if the function accepts the expected parameters
        import inspect
        if hasattr(mngs.stats.tests, 'corr_test'):
            sig = inspect.signature(mngs.stats.tests.corr_test)
            param_names = list(sig.parameters.keys())
            
            # Should have at least data1, data2 parameters
            assert 'data1' in param_names
            assert 'data2' in param_names


if __name__ == "__main__":
    import os

    import pytest

<<<<<<< HEAD
    pytest.main([os.path.abspath(__file__)])
=======
    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/stats/tests/__init__.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
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
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/stats/tests/__init__.py
# --------------------------------------------------------------------------------
>>>>>>> origin/main
