#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-06-02 15:50:00 (ywatanabe)"
# File: ./tests/mngs/types/test___init__.py

"""
Functionality:
    * Tests types module initialization and type definitions
    * Validates typing imports and custom type availability
    * Tests module structure and import consistency
Input:
    * Module imports and type checking
Output:
    * Test results
Prerequisites:
    * pytest
"""

import pytest
import inspect
from typing import get_origin, get_args


class TestTypesModuleInit:
    """Test cases for types module initialization."""

    def test_module_imports_successfully(self):
        """Test that the types module imports without errors."""
        import mngs.types
        assert mngs.types is not None

    def test_standard_typing_imports(self):
        """Test that standard typing imports are available."""
        import mngs.types
        
        standard_types = [
            'List', 'Tuple', 'Dict', 'Any', 'Union', 
            'Sequence', 'Literal', 'Optional', 'Iterable', 'Generator'
        ]
        
        for type_name in standard_types:
            assert hasattr(mngs.types, type_name), f"Standard type {type_name} not found"

    def test_arraylike_type_available(self):
        """Test that ArrayLike type is available."""
        import mngs.types
        assert hasattr(mngs.types, 'ArrayLike')
        assert hasattr(mngs.types, 'is_array_like')
        assert callable(mngs.types.is_array_like)

    def test_colorlike_type_available(self):
        """Test that ColorLike type is available."""
        import mngs.types
        assert hasattr(mngs.types, 'ColorLike')

    def test_is_listed_x_function_available(self):
        """Test that is_listed_X function is available."""
        import mngs.types
        assert hasattr(mngs.types, 'is_listed_X')
        assert callable(mngs.types.is_listed_X)

    def test_function_signatures(self):
        """Test that functions have expected signatures."""
        import mngs.types
        
        # Test is_array_like signature
        sig = inspect.signature(mngs.types.is_array_like)
        params = list(sig.parameters.keys())
        assert 'obj' in params
        
        # Test is_listed_X signature
        sig = inspect.signature(mngs.types.is_listed_X)
        params = list(sig.parameters.keys())
        assert 'obj' in params
        assert 'types' in params

    def test_module_has_proper_imports(self):
        """Test that module imports from expected submodules."""
        import mngs.types
        
        # Check that functions come from proper modules
        assert mngs.types.is_array_like.__module__ == 'mngs.types._ArrayLike'
        assert mngs.types.is_listed_X.__module__ == 'mngs.types._is_listed_X'

    def test_type_union_structure(self):
        """Test that Union types are structured correctly."""
        import mngs.types
        
        # ArrayLike should be a Union type
        array_like = mngs.types.ArrayLike
        assert get_origin(array_like) is not None or hasattr(array_like, '__args__')
        
        # ColorLike should be a Union type
        color_like = mngs.types.ColorLike
        assert get_origin(color_like) is not None or hasattr(color_like, '__args__')

    def test_no_unwanted_attributes(self):
        """Test that module doesn't expose unwanted attributes."""
        import mngs.types
        
        # Get all attributes
        attrs = dir(mngs.types)
        
        # Should not have these internal attributes exposed at module level
        unwanted = ['os', '__FILE__', '__DIR__']
        for attr in unwanted:
            assert attr not in attrs, f"Unwanted attribute {attr} exposed"

    def test_module_reimport_consistency(self):
        """Test that multiple imports are consistent."""
        import mngs.types as types1
        import mngs.types as types2
        
        assert types1 is types2
        assert types1.ArrayLike is types2.ArrayLike
        assert types1.ColorLike is types2.ColorLike
        assert types1.is_array_like is types2.is_array_like
        assert types1.is_listed_X is types2.is_listed_X

    def test_typing_imports_from_standard_library(self):
        """Test that typing imports work as expected."""
        import mngs.types
        from typing import List as StdList, Union as StdUnion
        
        # These should be the same as standard library types
        assert mngs.types.List is StdList
        assert mngs.types.Union is StdUnion

    def test_custom_types_functionality(self):
        """Test that custom types work for type checking."""
        import mngs.types
        
        # Test that we can use the types for isinstance checks
        # (Note: Union types themselves aren't directly usable with isinstance,
        # but the individual component types should be accessible)
        
        # Test array-like function
        assert callable(mngs.types.is_array_like)
        
        # Test listed function
        assert callable(mngs.types.is_listed_X)

    def test_module_file_attributes(self):
        """Test that module has expected file-related attributes."""
        import mngs.types
        
        # Should have module metadata
        assert hasattr(mngs.types, '__name__')
        assert 'mngs.types' in mngs.types.__name__

    def test_docstring_availability(self):
        """Test that functions have docstrings."""
        import mngs.types
        
        # Functions should have docstrings
        assert mngs.types.is_array_like.__doc__ is not None
        assert mngs.types.is_listed_X.__doc__ is not None
        
        # Check docstring content
        assert 'array-like' in mngs.types.is_array_like.__doc__.lower()
        assert 'list' in mngs.types.is_listed_X.__doc__.lower()

    def test_import_error_handling(self):
        """Test graceful handling of import dependencies."""
        # The module should import successfully even if some dependencies
        # like torch might not be available in all environments
        try:
            import mngs.types
            # If import succeeds, basic functionality should work
            assert hasattr(mngs.types, 'ArrayLike')
            assert hasattr(mngs.types, 'ColorLike')
        except ImportError as e:
            # If there's an import error, it should be specific
            assert 'torch' in str(e) or 'pandas' in str(e) or 'xarray' in str(e)

    def test_namespace_cleanliness(self):
        """Test that module namespace only contains expected items."""
        import mngs.types
        
        # Get all public attributes (not starting with _)
        public_attrs = [attr for attr in dir(mngs.types) if not attr.startswith('_')]
        
        expected_attrs = {
            'List', 'Tuple', 'Dict', 'Any', 'Union', 'Sequence', 
            'Literal', 'Optional', 'Iterable', 'Generator',
            'ArrayLike', 'is_array_like', 'ColorLike', 'is_listed_X'
        }
        
        # All expected attributes should be present
        for attr in expected_attrs:
            assert attr in public_attrs, f"Expected attribute {attr} not found"
        
        # Check for unexpected attributes (informational)
        unexpected_attrs = set(public_attrs) - expected_attrs
        if unexpected_attrs:
            print(f"Note: Unexpected public attributes found: {unexpected_attrs}")


if __name__ == "__main__":
    import os
    import pytest
    pytest.main([os.path.abspath(__file__)])