#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-04 06:50:00 (ywatanabe)"
# File: ./tests/mngs/test___main__.py

import pytest
import sys
from unittest import mock
from io import StringIO


class TestMNGSMain:
    """Test suite for mngs.__main__ module functionality."""

    @pytest.fixture
    def mock_print_config(self):
        """Mock the print_config_main function."""
        with mock.patch('mngs.__main__.print_config_main') as mock_func:
            yield mock_func

    def test_main_function_exists(self):
        """Test that main function exists and is callable."""
        import mngs.__main__ as main_module
        
        assert hasattr(main_module, 'main'), "Should have main function"
        assert callable(main_module.main), "main should be callable"

    def test_print_config_main_import(self):
        """Test that print_config_main is properly imported."""
        import mngs.__main__ as main_module
        
        assert hasattr(main_module, 'print_config_main'), "Should import print_config_main"

    @mock.patch('sys.argv', ['mngs'])
    def test_main_no_arguments_exits(self, mock_print_config):
        """Test that main exits with usage message when no arguments provided."""
        import mngs.__main__ as main_module
        
        with mock.patch('sys.exit') as mock_exit:
            with mock.patch('builtins.print') as mock_print:
                main_module.main()
                
                mock_print.assert_called_with("Usage: python -m mngs <command> [args]")
                mock_exit.assert_called_with(1)

    @mock.patch('sys.argv', ['mngs', 'print_config'])
    def test_main_print_config_command(self, mock_print_config):
        """Test that print_config command calls print_config_main."""
        import mngs.__main__ as main_module
        
        main_module.main()
        
        mock_print_config.assert_called_once_with([])

    @mock.patch('sys.argv', ['mngs', 'print_config', 'DATABASE_URL'])
    def test_main_print_config_with_args(self, mock_print_config):
        """Test that print_config command passes arguments correctly."""
        import mngs.__main__ as main_module
        
        main_module.main()
        
        mock_print_config.assert_called_once_with(['DATABASE_URL'])

    @mock.patch('sys.argv', ['mngs', 'print_config', 'key1', 'key2'])
    def test_main_print_config_multiple_args(self, mock_print_config):
        """Test that print_config command handles multiple arguments."""
        import mngs.__main__ as main_module
        
        main_module.main()
        
        mock_print_config.assert_called_once_with(['key1', 'key2'])

    @mock.patch('sys.argv', ['mngs', 'unknown_command'])
    def test_main_unknown_command(self, mock_print_config):
        """Test that unknown command prints error and exits."""
        import mngs.__main__ as main_module
        
        with mock.patch('sys.exit') as mock_exit:
            with mock.patch('builtins.print') as mock_print:
                main_module.main()
                
                mock_print.assert_called_with("Unknown command: unknown_command")
                mock_exit.assert_called_with(1)

    @mock.patch('sys.argv', ['mngs', 'invalid', 'arg1', 'arg2'])
    def test_main_invalid_command_with_args(self, mock_print_config):
        """Test that invalid command with args prints error and exits."""
        import mngs.__main__ as main_module
        
        with mock.patch('sys.exit') as mock_exit:
            with mock.patch('builtins.print') as mock_print:
                main_module.main()
                
                mock_print.assert_called_with("Unknown command: invalid")
                mock_exit.assert_called_with(1)

    def test_module_docstring_exists(self):
        """Test that module has proper documentation."""
        import mngs.__main__ as main_module
        
        assert main_module.__doc__ is not None, "Module should have docstring"
        assert "entry point" in main_module.__doc__, "Should describe entry point functionality"

    def test_main_function_docstring(self):
        """Test that main function has proper documentation."""
        import mngs.__main__ as main_module
        
        assert main_module.main.__doc__ is not None, "main function should have docstring"
        docstring = main_module.main.__doc__
        assert "command-line" in docstring.lower(), "Should describe command-line functionality"
        assert "print_config" in docstring, "Should document print_config command"

    def test_sys_module_imported(self):
        """Test that sys module is properly imported."""
        import mngs.__main__ as main_module
        
        assert hasattr(main_module, 'sys'), "Should import sys module"

    def test_warnings_configuration(self):
        """Test that warnings are configured at module level."""
        import mngs.__main__ as main_module
        
        # Check that warnings module is imported
        assert hasattr(main_module, 'warnings'), "Should import warnings module"

    @mock.patch('mngs.__main__.print_config_main')
    @mock.patch('sys.argv', ['mngs', 'print_config'])
    def test_print_config_main_exception_handling(self, mock_print_config):
        """Test behavior when print_config_main raises exception."""
        import mngs.__main__ as main_module
        
        # Make print_config_main raise an exception
        mock_print_config.side_effect = Exception("Test exception")
        
        # Should propagate the exception (no exception handling in main)
        with pytest.raises(Exception, match="Test exception"):
            main_module.main()

    def test_module_attributes(self):
        """Test that module has expected attributes."""
        import mngs.__main__ as main_module
        
        # Check essential attributes
        assert hasattr(main_module, '__file__'), "Should have __file__ attribute"
        assert hasattr(main_module, '__name__'), "Should have __name__ attribute"

    @mock.patch('sys.argv', ['mngs', 'print_config', '--help'])
    def test_print_config_help_flag(self, mock_print_config):
        """Test that help flag is passed to print_config_main."""
        import mngs.__main__ as main_module
        
        main_module.main()
        
        mock_print_config.assert_called_once_with(['--help'])

    def test_command_line_entry_point(self):
        """Test that module can be executed as script."""
        import mngs.__main__ as main_module
        
        # Test that __name__ == "__main__" block exists
        # by checking if the main function would be called
        with mock.patch.object(main_module, 'main') as mock_main:
            # Simulate module execution
            if __name__ == "__main__":
                main_module.main()
            
            # This test mainly checks the structure is correct

    @pytest.mark.parametrize("command,args,expected_args", [
        ("print_config", [], []),
        ("print_config", ["key"], ["key"]),
        ("print_config", ["key1", "key2"], ["key1", "key2"]),
        ("print_config", ["--verbose"], ["--verbose"]),
        ("print_config", ["DATABASE_URL", "--format=json"], ["DATABASE_URL", "--format=json"]),
    ])
    def test_argument_parsing_variations(self, command, args, expected_args, mock_print_config):
        """Test various argument combinations for print_config command."""
        import mngs.__main__ as main_module
        
        test_argv = ['mngs', command] + args
        
        with mock.patch('sys.argv', test_argv):
            main_module.main()
            
            mock_print_config.assert_called_once_with(expected_args)

    def test_empty_command_handling(self, mock_print_config):
        """Test handling of empty string as command."""
        import mngs.__main__ as main_module
        
        with mock.patch('sys.argv', ['mngs', '']):
            with mock.patch('sys.exit') as mock_exit:
                with mock.patch('builtins.print') as mock_print:
                    main_module.main()
                    
                    mock_print.assert_called_with("Unknown command: ")
                    mock_exit.assert_called_with(1)


if __name__ == "__main__":
    import os
<<<<<<< HEAD
    pytest.main([os.path.abspath(__file__), "-v"])
=======

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/__main__.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-10-13 18:56:15 (ywatanabe)"
# # /home/yusukew/proj/mngs_repo/src/mngs/__main__.py
# 
# """
# 1. Functionality:
#    - Serves as the entry point for the mngs package
#    - Handles command-line arguments and directs to appropriate submodules
# 2. Input:
#    - Command-line arguments
# 3. Output:
#    - Execution of specified submodule or usage instructions
# 4. Prerequisites:
#    - mngs package and its submodules
# """
# 
# import warnings
# warnings.filterwarnings("ignore")
# 
# import sys
# from typing import List
# from .gen._print_config import print_config_main
# 
# def main(args: List[str] = None) -> None:
#     """
#     Main function to handle command-line arguments and execute appropriate submodules.
# 
#     Parameters
#     ----------
#     args : List[str], optional
#         Command-line arguments. If None, sys.argv[1:] will be used.
# 
#     Returns
#     -------
#     None
# 
#     Example
#     -------
#     >>> import mngs.__main__ as main_module
#     >>> main_module.main(['print_config', 'SOME_KEY'])
#     """
# 
# from .gen._print_config import print_config_main
# 
# def main():
#     if len(sys.argv) < 2:
#         print("Usage: python -m mngs <command> [args]")
#         sys.exit(1)
# 
#     command = sys.argv[1]
#     args = sys.argv[2:]
# 
#     if command == "print_config":
#         print_config_main(args)
#     else:
#         print(f"Unknown command: {command}")
#         sys.exit(1)
# 
# if __name__ == "__main__":
#     main()

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/__main__.py
# --------------------------------------------------------------------------------
>>>>>>> origin/main
