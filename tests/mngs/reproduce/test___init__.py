#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-06-02 15:40:00 (ywatanabe)"
# File: ./mngs_repo/tests/mngs/reproduce/test___init__.py

"""Tests for reproduce module __init__.py."""

import pytest
import mngs.reproduce


def test_reproduce_module_imports():
    """Test that reproduce module imports all expected functions."""
    # Check that expected functions are available
    expected_functions = [
        'gen_id',      # from _gen_ID.py
        'gen_ID',      # backward compatibility alias
        'gen_timestamp',  # from _gen_timestamp.py
        'timestamp',   # alias
        'fix_seeds',   # from _fix_seeds.py
    ]
    
    for func_name in expected_functions:
        assert hasattr(mngs.reproduce, func_name), f"Missing {func_name} in mngs.reproduce"


def test_no_private_functions_exposed():
    """Test that private functions are not exposed."""
    # Check expected public interface
    expected_public = ['gen_id', 'gen_ID', 'gen_timestamp', 'timestamp', 'fix_seeds']
    
    for attr_name in expected_public:
        assert hasattr(mngs.reproduce, attr_name), f"Missing public attribute {attr_name}"


def test_imported_functions_are_callable():
    """Test that imported items are callable functions."""
    import inspect
    
    # Get public attributes
    public_attrs = [attr for attr in dir(mngs.reproduce) 
                   if not attr.startswith('_') and hasattr(mngs.reproduce, attr)]
    
    for attr_name in public_attrs:
        attr = getattr(mngs.reproduce, attr_name)
        # Should be functions (not classes in this module)
        assert callable(attr), f"{attr_name} should be callable"


def test_gen_id_functionality():
    """Test gen_id works when imported from mngs.reproduce."""
    # Basic test
    id1 = mngs.reproduce.gen_id()
    assert isinstance(id1, str)
    assert '_' in id1
    
    # Custom parameters
    id2 = mngs.reproduce.gen_id(N=4)
    parts = id2.split('_')
    assert len(parts[1]) == 4


def test_gen_timestamp_functionality():
    """Test gen_timestamp works when imported from mngs.reproduce."""
    ts = mngs.reproduce.gen_timestamp()
    assert isinstance(ts, str)
    assert len(ts) == 14  # YYYY-MMDD-HHMM format
    
    # Also test alias
    ts2 = mngs.reproduce.timestamp()
    assert isinstance(ts2, str)
    assert len(ts2) == 14


def test_fix_seeds_functionality():
    """Test fix_seeds works when imported from mngs.reproduce."""
    import random
    import numpy as np
    
    # Fix seeds - need to pass modules as parameters
    mngs.reproduce.fix_seeds(random=random, np=np, seed=42, verbose=False)
    
    # Test Python random
    val1 = random.random()
    
    # Fix seeds again with same seed
    mngs.reproduce.fix_seeds(random=random, np=np, seed=42, verbose=False)
    val2 = random.random()
    
    # Should produce same values
    assert val1 == val2
    
    # Test numpy
    mngs.reproduce.fix_seeds(np=np, seed=123, verbose=False)
    arr1 = np.random.rand(5)
    
    mngs.reproduce.fix_seeds(np=np, seed=123, verbose=False)
    arr2 = np.random.rand(5)
    
    assert np.array_equal(arr1, arr2)


def test_backward_compatibility():
    """Test backward compatibility aliases."""
    # gen_ID should be available and same as gen_id
    assert hasattr(mngs.reproduce, 'gen_ID')
    assert mngs.reproduce.gen_ID is mngs.reproduce.gen_id
    
    # timestamp should be available and same as gen_timestamp
    assert hasattr(mngs.reproduce, 'timestamp')
    assert mngs.reproduce.timestamp is mngs.reproduce.gen_timestamp


def test_no_import_side_effects():
    """Test that importing doesn't have side effects."""
    import importlib
    
    # Re-import module
    importlib.reload(mngs.reproduce)
    
    # Should still have all functions
    assert hasattr(mngs.reproduce, 'gen_id')
    assert hasattr(mngs.reproduce, 'gen_timestamp')
    assert hasattr(mngs.reproduce, 'fix_seeds')


def test_module_documentation():
    """Test that imported functions retain documentation."""
    # Check docstrings exist
    assert mngs.reproduce.gen_id.__doc__ is not None
    assert mngs.reproduce.gen_timestamp.__doc__ is not None
    # Note: fix_seeds has no docstring in source
    
    # Check they contain expected content
    assert "unique identifier" in mngs.reproduce.gen_id.__doc__.lower()
    assert "timestamp" in mngs.reproduce.gen_timestamp.__doc__.lower()


def test_no_temporary_variables():
    """Test that temporary import variables are cleaned up."""
    # These should not exist in module namespace
    assert not hasattr(mngs.reproduce, 'os')
    assert not hasattr(mngs.reproduce, 'importlib')
    assert not hasattr(mngs.reproduce, 'inspect')
    assert not hasattr(mngs.reproduce, 'current_dir')
    assert not hasattr(mngs.reproduce, 'filename')
    assert not hasattr(mngs.reproduce, 'module_name')
    assert not hasattr(mngs.reproduce, 'module')
    assert not hasattr(mngs.reproduce, 'name')
    assert not hasattr(mngs.reproduce, 'obj')


def test_function_signatures():
    """Test function signatures are preserved."""
    import inspect
    
    # Test gen_id signature
    sig = inspect.signature(mngs.reproduce.gen_id)
    params = list(sig.parameters.keys())
    assert 'time_format' in params
    assert 'N' in params
    
    # Test gen_timestamp signature (no parameters)
    sig = inspect.signature(mngs.reproduce.gen_timestamp)
    assert len(sig.parameters) == 0
    
    # Test fix_seeds signature
    sig = inspect.signature(mngs.reproduce.fix_seeds)
    params = list(sig.parameters.keys())
    assert 'seed' in params
    assert 'np' in params
    assert 'torch' in params


def test_all_functions_from_submodules():
    """Test all public functions from submodules are available."""
    # Import submodules directly to compare
    from mngs.reproduce import _gen_ID, _gen_timestamp, _fix_seeds
    
    # Check gen_id/gen_ID
    assert hasattr(_gen_ID, 'gen_id')
    assert hasattr(mngs.reproduce, 'gen_id')
    
    # Check gen_timestamp/timestamp
    assert hasattr(_gen_timestamp, 'gen_timestamp')
    assert hasattr(mngs.reproduce, 'gen_timestamp')
    
    # Check fix_seeds
    assert hasattr(_fix_seeds, 'fix_seeds')
    assert hasattr(mngs.reproduce, 'fix_seeds')


def test_reproducibility_workflow():
    """Test typical reproducibility workflow using the module."""
    import numpy as np
    import random
    
    # Set seeds for reproducibility
    mngs.reproduce.fix_seeds(random=random, np=np, seed=42, verbose=False)
    
    # Generate reproducible ID (note: need to reset seed for random part)
    random.seed(42)  # Reset for reproducible random part
    exp_id = mngs.reproduce.gen_id(N=6)
    assert len(exp_id.split('_')[1]) == 6
    
    # Generate timestamp
    ts = mngs.reproduce.gen_timestamp()
    assert len(ts) == 14
    
    # Create reproducible random data
    mngs.reproduce.fix_seeds(np=np, seed=42, verbose=False)
    data1 = np.random.rand(10)
    
    # Reset seeds and verify reproducibility
    mngs.reproduce.fix_seeds(np=np, seed=42, verbose=False)
    data2 = np.random.rand(10)
    
    assert np.array_equal(data1, data2)


if __name__ == "__main__":
    import os
    pytest.main([os.path.abspath(__file__), "-v"])


<<<<<<< HEAD
# EOF
=======
    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/reproduce/__init__.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 14:29:30 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/reproduce/__init__.py
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
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/reproduce/__init__.py
# --------------------------------------------------------------------------------
>>>>>>> origin/main
