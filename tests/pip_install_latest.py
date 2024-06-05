#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-06-05 10:24:12 (ywatanabe)"
# /home/ywatanabe/proj/mngs/tests/pip_install_latest.py

import os
import sys

import requests


def get_latest_release_tag(repo):
    """Fetch the latest release tag from a GitHub repository."""
    url = f"https://api.github.com/repos/{repo}/tags"
    response = requests.get(url)
    tags = response.json()
    if not tags:
        return None
    # Assuming the first tag in the list is the latest
    latest_tag = tags[0]["name"]
    return latest_tag


def install_package(repo, tag):
    """Install a package from GitHub using pip and a specific tag."""
    command = f"pip install git+https://github.com/{repo}@{tag}"
    print(f"Executing: {command}")
    os.system(command)


def main():
    if len(sys.argv) != 2:
        print(
            "\nExample usage:\npython pip_install_latest.py ywatanabe1989/mngs"
        )

        sys.exit(1)

    repo = sys.argv[1]
    latest_tag = get_latest_release_tag(repo)
    if latest_tag:
        print(f"Installing {repo} at tag {latest_tag}")
        install_package(repo, latest_tag)
    else:
        print("No tags found for the repository.")


if __name__ == "__main__":
    main()

# EOF
