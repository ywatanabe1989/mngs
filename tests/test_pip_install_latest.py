#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-11 08:58:32 (ywatanabe)"
# /home/ywatanabe/proj/mngs/tests/test_pip_install_latest.py

import logging
from unittest.mock import MagicMock, patch

import pytest
from pip_install_latest import get_latest_release_tag, install_package, main


@pytest.fixture
def mock_requests_get():
    with patch("requests.get") as mock_get:
        yield mock_get


@pytest.fixture
def mock_subprocess_call():
    with patch("subprocess.call") as mock_call:
        yield mock_call


def test_get_latest_release_tag_success(mock_requests_get):
    mock_response = MagicMock()
    mock_response.json.return_value = [{"name": "v1.0.0"}]
    mock_requests_get.return_value = mock_response

    result = get_latest_release_tag("test/repo")
    assert result == "v1.0.0"


def test_get_latest_release_tag_no_tags(mock_requests_get):
    mock_response = MagicMock()
    mock_response.json.return_value = []
    mock_requests_get.return_value = mock_response

    result = get_latest_release_tag("test/repo")
    assert result is None


def test_install_package_success(mock_subprocess_call):
    mock_subprocess_call.return_value = 0
    result = install_package("test/repo", "v1.0.0")
    assert result == 0
    mock_subprocess_call.assert_called_once_with(
        "pip install git+https://github.com/test/repo@v1.0.0", shell=True
    )


def test_install_package_failure(mock_subprocess_call):
    mock_subprocess_call.return_value = 1
    result = install_package("test/repo", "v1.0.0")
    assert result == 1


@pytest.mark.parametrize(
    "tag,expected_log",
    [
        ("v1.0.0", "Installation successful"),
        (None, "No tags found for the repository."),
    ],
)
def test_main(
    tag, expected_log, mock_requests_get, mock_subprocess_call, caplog
):
    caplog.set_level(logging.INFO)

    mock_response = MagicMock()
    mock_response.json.return_value = [{"name": tag}] if tag else []
    mock_requests_get.return_value = mock_response
    mock_subprocess_call.return_value = 0 if tag else 1

    with patch("sys.argv", ["script_name", "test/repo"]):
        with patch("logging.basicConfig") as mock_logging:
            main()

    assert expected_log in caplog.text

    # Additional debugging
    print(f"Captured logs: {caplog.text}")
    print(f"Mock calls: {mock_logging.call_args_list}")


# EOF

# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-06-05 10:24:12 (ywatanabe)"
# # /home/ywatanabe/proj/mngs/tests/pip_install_latest.py

# import os
# import sys

# import requests


# def get_latest_release_tag(repo):
#     """Fetch the latest release tag from a GitHub repository."""
#     url = f"https://api.github.com/repos/{repo}/tags"
#     response = requests.get(url)
#     tags = response.json()
#     if not tags:
#         return None
#     # Assuming the first tag in the list is the latest
#     latest_tag = tags[0]["name"]
#     return latest_tag


# def install_package(repo, tag):
#     """Install a package from GitHub using pip and a specific tag."""
#     command = f"pip install git+https://github.com/{repo}@{tag}"
#     print(f"Executing: {command}")
#     os.system(command)


# def main():
#     if len(sys.argv) != 2:
#         print(
#             "\nExample usage:\npython pip_install_latest.py ywatanabe1989/mngs"
#         )

#         sys.exit(1)

#     repo = sys.argv[1]
#     latest_tag = get_latest_release_tag(repo)
#     if latest_tag:
#         print(f"Installing {repo} at tag {latest_tag}")
#         install_package(repo, latest_tag)
#     else:
#         print("No tags found for the repository.")


# if __name__ == "__main__":
#     main()

# # EOF
