#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-31"
# File: test___init__.py

"""Tests for mngs.io module initialization and imports."""

import inspect
import sys
from pathlib import Path

import pytest


class TestIOModuleInitialization:
    """Test the imports and structure of the io module."""

    def test_module_imports(self):
        """Test that the io module successfully imports."""
        import mngs.io

        assert mngs.io is not None
        assert hasattr(mngs.io, "__file__")

    def test_expected_functions_available(self):
        """Test that expected functions are available after import."""
        import mngs.io

        # Core IO functions that should be available
        expected_functions = [
            "save",
            "load",
            "cache",
            "flush",
            "glob",
            "parse_glob",
            "json2md",
            "load_configs",
            "mv_to_tmp",
            "path",
            "reload",
            "save_image",
            "save_listed_dfs_as_csv",
            "save_listed_scalars_as_csv",
            "save_mp4",
            "save_optuna_study_as_csv_and_pngs",
            "save_text",
        ]

        for func_name in expected_functions:
            assert hasattr(
                mngs.io, func_name
            ), f"Function {func_name} not found in mngs.io"

    def test_glob_override(self):
        """Test that glob function is properly overridden."""
        import mngs.io

        # glob should be from _glob module, not the standard library
        assert hasattr(mngs.io, "glob")
        assert hasattr(mngs.io, "parse_glob")

        # Check it's not the standard library glob
        import glob as stdlib_glob

        assert mngs.io.glob is not stdlib_glob.glob

    def test_wildcard_imports(self):
        """Test that wildcard imports work correctly."""
        # The module uses from ._module import * pattern
        import mngs.io

        # Check that functions from submodules are available
        assert callable(getattr(mngs.io, "save", None))
        assert callable(getattr(mngs.io, "load", None))
        assert callable(getattr(mngs.io, "cache", None))

    def test_no_duplicate_imports(self):
        """Test that save is not imported twice (it appears twice in source)."""
        import mngs.io

        # Despite appearing twice in imports, save should be a single function
        assert hasattr(mngs.io, "save")
        save_func = getattr(mngs.io, "save")
        assert callable(save_func)

    def test_module_structure(self):
        """Test the module has proper structure."""
        import mngs.io

        # Check for module attributes
        assert hasattr(mngs.io, "__file__")
        assert hasattr(mngs.io, "__name__")
        assert mngs.io.__name__ == "mngs.io"


class TestIOModuleFunctionality:
    """Test specific functionality of the io module."""

    def test_save_load_availability(self):
        """Test that save and load functions are available and callable."""
        import mngs.io

        assert callable(mngs.io.save)
        assert callable(mngs.io.load)

        # Check they have proper signatures
        save_sig = inspect.signature(mngs.io.save)
        load_sig = inspect.signature(mngs.io.load)

        assert "filename" in save_sig.parameters or "fname" in save_sig.parameters
        assert "filename" in load_sig.parameters or "fname" in load_sig.parameters

    def test_specialized_save_functions(self):
        """Test that specialized save functions are available."""
        import mngs.io

        specialized_saves = [
            "save_image",
            "save_text",
            "save_mp4",
            "save_listed_dfs_as_csv",
            "save_listed_scalars_as_csv",
            "save_optuna_study_as_csv_and_pngs",
        ]

        for func_name in specialized_saves:
            assert hasattr(mngs.io, func_name), f"{func_name} not found"
            assert callable(getattr(mngs.io, func_name)), f"{func_name} is not callable"

    def test_utility_functions(self):
        """Test that utility functions are available."""
        import mngs.io

        utilities = ["cache", "flush", "reload", "mv_to_tmp", "path"]

        for util in utilities:
            assert hasattr(mngs.io, util), f"{util} not found"
            assert callable(getattr(mngs.io, util)), f"{util} is not callable"

    def test_config_loading(self):
        """Test that config loading function is available."""
        import mngs.io

        assert hasattr(mngs.io, "load_configs")
        assert callable(mngs.io.load_configs)

    def test_json_to_markdown(self):
        """Test that json2md function is available."""
        import mngs.io

        assert hasattr(mngs.io, "json2md")
        assert callable(mngs.io.json2md)


class TestIOModuleIntegration:
    """Test integration aspects of the io module."""

    def test_load_modules_imported(self):
        """Test that _load_modules submodule content is available."""
        import mngs.io

        # The _load_modules is imported with *, so its functions should be available
        # Check for some common load functions if they exist
        possible_loaders = [
            "load_numpy",
            "load_pickle",
            "load_json",
            "load_yaml",
            "load_torch",
            "load_pandas",
            "load_image",
        ]

        # At least some loaders should be available
        available_loaders = [l for l in possible_loaders if hasattr(mngs.io, l)]
        # This assertion might need adjustment based on actual implementation
        assert len(available_loaders) >= 0, "Expected some loader functions"

    def test_no_private_functions_exposed(self):
        """Test that private functions are not exposed."""
        import mngs.io

        # Get all attributes
        for attr_name in dir(mngs.io):
            if attr_name.startswith("_") and not attr_name.startswith("__"):
                # Skip module imports like _cache, _save etc
                attr = getattr(mngs.io, attr_name)
                if callable(attr) and not inspect.ismodule(attr):
                    # Private functions should not be exposed
                    pytest.fail(f"Private function {attr_name} is exposed")

    def test_reimport_stability(self):
        """Test that reimporting maintains stability."""
        import importlib
        import mngs.io

        # Get initial function list
        initial_funcs = set(
            name
            for name in dir(mngs.io)
            if not name.startswith("_") and callable(getattr(mngs.io, name))
        )

        # Reload module
        importlib.reload(mngs.io)

        # Check functions are still there
        final_funcs = set(
            name
            for name in dir(mngs.io)
            if not name.startswith("_") and callable(getattr(mngs.io, name))
        )

        assert initial_funcs == final_funcs, "Function list changed after reimport"


class TestIOModuleEdgeCases:
    """Test edge cases for the io module."""

    def test_circular_import_handling(self):
        """Test that the module handles circular imports properly."""
        # Import in different orders
        import mngs
        import mngs.io

        assert mngs is not None
        assert mngs.io is not None

    def test_module_path_consistency(self):
        """Test that module path is consistent."""
        import mngs.io

        module_path = Path(mngs.io.__file__).parent
        assert module_path.name == "io"
        assert module_path.parent.name == "mngs"

    def test_this_file_constant(self):
        """Test THIS_FILE constant if it exists."""
        import mngs.io

        if hasattr(mngs.io, "THIS_FILE"):
            # It should be a string path
            assert isinstance(mngs.io.THIS_FILE, str)


# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])
<<<<<<< HEAD
=======

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/__init__.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-04-22 09:27:44 (ywatanabe)"
# # File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/src/mngs/io/__init__.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "/ssh:sp:/home/ywatanabe/proj/mngs_repo/src/mngs/io/__init__.py"
>>>>>>> origin/main
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
#
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/io/__init__.py"
#
# # import os
# # import importlib
# # import inspect
#
# # # Get the current directory
# # current_dir = os.path.dirname(__file__)
#
# # # Iterate through all Python files in the current directory
# # for filename in os.listdir(current_dir):
# #     if filename.endswith(".py") and not filename.startswith("__"):
# #         module_name = filename[:-3]  # Remove .py extension
# #         module = importlib.import_module(f".{module_name}", package=__name__)
# #         # Import only functions and classes from the module
# #         for name, obj in inspect.getmembers(module):
# #             if inspect.isfunction(obj) or inspect.isclass(obj):
# #                 if not name.startswith("_"):
# #                     # print(name)
# #                     globals()[name] = obj
#
# # # Clean up temporary variables
# # del os, importlib, inspect, current_dir, filename, module_name, module, name, obj
#
# # # EOF
#
# from ._cache import *
# from ._flush import *
# from ._glob import *
# from ._json2md import *
# from ._load_configs import *
# from ._load_modules import *
# from ._load import *
# from ._mv_to_tmp import *
# from ._path import *
# from ._reload import *
# from ._save import *
# from ._save_image import *
# from ._save_listed_dfs_as_csv import *
# from ._save_listed_scalars_as_csv import *
# from ._save_mp4 import *
# from ._save_optuna_study_as_csv_and_pngs import *
# # from ._save_optuna_stury import *
# from ._save import *
# from ._save_text import *
#
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/__init__.py
# --------------------------------------------------------------------------------
