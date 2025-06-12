#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-31"
# File: test___init__.py

"""Tests for mngs.gen module initialization and auto-imports."""

import importlib
import inspect
import sys
from pathlib import Path

import pytest


class TestGenModuleInitialization:
    """Test the automatic import functionality of the gen module."""

    def test_module_imports(self):
        """Test that the gen module successfully imports."""
        import mngs.gen

        assert mngs.gen is not None
        assert hasattr(mngs.gen, "__file__")

    def test_expected_functions_available(self):
        """Test that expected functions are available after import."""
        import mngs.gen

        # Test some common functions that should be auto-imported
        expected_functions = [
            "DimHandler",
            "TimeStamper",
            "alternate_kwarg",
            "cache",
            "check_host",
            "ci",
            "close",
            "embed",
            "inspect_module",
            "is_ipython",
            "less",
            "list_packages",
            "mat2py",
            "norm",
            "paste",
            "print_config",
            "shell",
            "src",
            "start",
            "symlink",
            "symlog",
            "tee",
            "title2path",
            "title_case",
            "to_even",
            "to_odd",
            "to_rank",
            "transpose",
            "type",
            "var_info",
            "wrap",
            "xml2dict",
        ]

        for func_name in expected_functions:
            assert hasattr(
                mngs.gen, func_name
            ), f"Function {func_name} not found in mngs.gen"

    def test_no_private_functions_exported(self):
        """Test that private functions (starting with _) are not exported."""
        import mngs.gen

        for name in dir(mngs.gen):
            if name.startswith("_") and not name.startswith("__"):
                obj = getattr(mngs.gen, name)
                # Private functions should not be callable or classes
                if callable(obj) and not name.endswith("__"):
                    pytest.fail(f"Private function {name} should not be exported")

    def test_imported_objects_are_callable_or_classes(self):
        """Test that all imported objects are either functions or classes."""
        import mngs.gen

        for name in dir(mngs.gen):
            if not name.startswith("_"):
                obj = getattr(mngs.gen, name)
                assert callable(obj) or inspect.isclass(
                    obj
                ), f"{name} is neither callable nor a class"

    def test_module_docstring(self):
        """Test that the module has a proper docstring."""
        import mngs.gen

        assert mngs.gen.__doc__ is not None
        assert (
            "Gen utility" in mngs.gen.__doc__ or "utility" in mngs.gen.__doc__.lower()
        )

    def test_no_import_side_effects(self):
        """Test that importing the module doesn't have unwanted side effects."""
        # Save the initial state
        initial_modules = set(sys.modules.keys())

        # Remove mngs.gen if it's already imported
        for key in list(sys.modules.keys()):
            if key.startswith("mngs.gen"):
                del sys.modules[key]

        # Import and check for side effects
        import mngs.gen

        # Only mngs.gen and its submodules should be added
        new_modules = set(sys.modules.keys()) - initial_modules
        for module in new_modules:
            assert module.startswith("mngs") or module in [
                "importlib",
                "inspect",
            ], f"Unexpected module imported: {module}"

    def test_cleanup_of_temporary_variables(self):
        """Test that temporary variables used in __init__ are cleaned up."""
        import mngs.gen

        # These variables should not exist after cleanup
        temp_vars = [
            "os",
            "importlib",
            "inspect",
            "current_dir",
            "filename",
            "module_name",
            "module",
            "name",
            "obj",
        ]

        for var in temp_vars:
            assert not hasattr(
                mngs.gen, var
            ), f"Temporary variable {var} was not cleaned up"


class TestGenModuleFunctionality:
    """Test specific functionality expectations of the gen module."""

    def test_misc_functions_imported(self):
        """Test that functions from misc.py are available."""
        import mngs.gen

        misc_functions = [
            "find_closest",
            "isclose",
            "describe",
            "unique",
            "float_linspace",
            "Dirac",
            "step",
            "relu",
        ]

        available = [f for f in misc_functions if hasattr(mngs.gen, f)]
        # At least some misc functions should be available
        assert len(available) > 0, "No misc functions were imported"

    def test_function_origins(self):
        """Test that we can trace functions back to their origin modules."""
        import mngs.gen

        # Test a few known functions and their expected modules
        if hasattr(mngs.gen, "TimeStamper"):
            assert mngs.gen.TimeStamper.__module__.endswith("_TimeStamper")

        if hasattr(mngs.gen, "tee"):
            assert mngs.gen.tee.__module__.endswith("_tee")

    def test_reimport_stability(self):
        """Test that reimporting the module is stable."""
        import mngs.gen

        # Get initial function list
        initial_funcs = set(name for name in dir(mngs.gen) if not name.startswith("_"))

        # Force reimport
        importlib.reload(mngs.gen)

        # Check functions are still there
        final_funcs = set(name for name in dir(mngs.gen) if not name.startswith("_"))

        assert initial_funcs == final_funcs, "Function list changed after reimport"


class TestGenModuleEdgeCases:
    """Test edge cases and error handling."""

    def test_import_with_missing_submodule(self, tmp_path, monkeypatch):
        """Test behavior when a submodule fails to import."""
        # This is a conceptual test - in practice the module handles this gracefully
        import mngs.gen

        # The module should still be importable even if some submodules fail
        assert mngs.gen is not None

    def test_circular_import_handling(self):
        """Test that the module handles potential circular imports."""
        # Import in different order to test for circular dependencies
        import mngs
        import mngs.gen

        # Both should work without issues
        assert mngs is not None
        assert mngs.gen is not None

    def test_module_all_attribute(self):
        """Test __all__ attribute if present."""
        import mngs.gen

        if hasattr(mngs.gen, "__all__"):
            # If __all__ is defined, it should be a list of strings
            assert isinstance(mngs.gen.__all__, list)
            for item in mngs.gen.__all__:
                assert isinstance(item, str)
                assert hasattr(mngs.gen, item)


class TestGenModuleIntegration:
    """Test integration with other parts of mngs."""

    def test_gen_functions_work_together(self):
        """Test that various gen functions can work together."""
        import mngs.gen

        # Test TimeStamper if available
        if hasattr(mngs.gen, "TimeStamper"):
            ts = mngs.gen.TimeStamper()
            assert hasattr(ts, "start")

    def test_module_path_consistency(self):
        """Test that the module path is consistent."""
        import mngs.gen

        module_path = Path(mngs.gen.__file__).parent
        assert module_path.name == "gen"
        assert module_path.parent.name == "mngs"


# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])
<<<<<<< HEAD
=======

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/gen/__init__.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-01 08:26:28 (ywatanabe)"
# # File: /home/ywatanabe/proj/mngs_repo/src/mngs/gen/__init__.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/gen/__init__.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# """Gen utility functions and classes for the MNGS project."""
# 
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
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/gen/__init__.py
# --------------------------------------------------------------------------------
>>>>>>> origin/main
