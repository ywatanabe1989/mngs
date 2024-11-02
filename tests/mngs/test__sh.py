#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-03 02:49:50 (ywatanabe)"
# File: ./mngs_repo/tests/mngs/test__sh.py

# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-03 02:23:16 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/_sh.py
#
# import subprocess
# import mngs
#
# def sh(command_str, verbose=True):
#     """
#     Executes a shell command from Python.
#
#     Parameters:
#     - command_str (str): The command string to execute.
#
#     Returns:
#     - output (str): The standard output from the executed command.
#     """
#     if verbose:
#         print(mngs.gen.color_text(f"{command_str}", "yellow"))
#
#     process = subprocess.Popen(
#         command_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
#     )
#     output, error = process.communicate()
#     if process.returncode == 0:
#         out = output.decode("utf-8").strip()
#     else:
#         out = error.decode("utf-8").strip()
#
#     if verbose:
#         print(out)
#
#     return out
#
#
# if __name__ == "__main__":
#     import sys
#
#     import matplotlib.pyplot as plt
#     import mngs
#
#     CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
#         sys, plt, verbose=False
#     )
#     sh("ls")
#     mngs.gen.close(CONFIG, verbose=False, notify=False)
#
#
# # EOF

# test from here --------------------------------------------------------------------------------

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.mngs._sh import sh

class TestShCommand:
    @pytest.fixture
    def mock_popen(self):
        with patch('subprocess.Popen') as mock:
            process_mock = MagicMock()
            process_mock.returncode = 0
            process_mock.communicate.return_value = (b'output', b'')
            mock.return_value = process_mock
            yield mock

    @pytest.fixture
    def mock_color_text(self):
        with patch('mngs.gen.color_text') as mock:
            mock.return_value = "colored_text"
            yield mock

    def test_successful_command(self, mock_popen, mock_color_text, capsys):
        """Tests successful command execution."""
        result = sh("ls", verbose=True)

        mock_popen.assert_called_once_with(
            "ls", shell=True, stdout=-1, stderr=-1
        )
        assert result == "output"

        captured = capsys.readouterr()
        assert "colored_text" in captured.out
        assert "output" in captured.out

    def test_failed_command(self, mock_popen, mock_color_text, capsys):
        """Tests failed command execution."""
        mock_popen.return_value.returncode = 1
        mock_popen.return_value.communicate.return_value = (b'', b'error')

        result = sh("invalid_command", verbose=True)

        assert result == "error"
        captured = capsys.readouterr()
        assert "colored_text" in captured.out
        assert "error" in captured.out

    def test_non_verbose_mode(self, mock_popen, mock_color_text, capsys):
        """Tests execution without verbose output."""
        result = sh("ls", verbose=False)

        mock_color_text.assert_not_called()
        captured = capsys.readouterr()
        assert captured.out == ""
        assert result == "output"

    def test_unicode_handling(self, mock_popen, capsys):
        """Tests handling of Unicode characters."""
        mock_popen.return_value.communicate.return_value = (
            "こんにちは".encode('utf-8'),
            b''
        )

        result = sh("echo こんにちは", verbose=True)
        assert result == "こんにちは"

    def test_empty_command(self, mock_popen):
        """Tests empty command handling."""
        with pytest.raises(subprocess.SubprocessError):
            sh("")

    @pytest.mark.parametrize("command,expected", [
        ("echo 'test'", "test"),
        ("pwd", "/current/dir"),
        ("whoami", "user"),
    ])
    def test_various_commands(self, mock_popen, command, expected):
        """Tests various command patterns."""
        mock_popen.return_value.communicate.return_value = (
            expected.encode('utf-8'),
            b''
        )
        result = sh(command, verbose=False)
        assert result == expected

# EOF
