#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-11 08:56:53 (ywatanabe)"
# /home/ywatanabe/proj/_mngs_repo_openhands/tests/pip_install_latest.py

import argparse
import logging
import os
import subprocess
import sys

import requests


def get_latest_release_tag(repository):
    """
    Fetch the latest release tag from a GitHub repository.

    Example
    -------
    repo = "ywatanabe1989/mngs"
    tag = get_latest_release_tag(repo)
    print(tag)

    Parameters
    ----------
    repository : str
        GitHub repository in the format "username/repository"

    Returns
    -------
    str or None
        Latest release tag if found, None otherwise
    """
    url = f"https://api.github.com/repos/{repository}/tags"
    response = requests.get(url)
    tags = response.json()
    return tags[0]["name"] if tags else None


def install_package(repository, tag):
    """
    Install a package from GitHub using pip and a specific tag.

    Example
    -------
    repo = "ywatanabe1989/mngs"
    tag = "v1.0.0"
    install_package(repo, tag)

    Parameters
    ----------
    repository : str
        GitHub repository in the format "username/repository"
    tag : str
        Tag to install

    Returns
    -------
    int
        Return code of the pip install command
    """
    command = f"pip install git+https://github.com/{repository}@{tag}"
    logging.info(f"Executing: {command}")
    return subprocess.call(command, shell=True)


def main():
    parser = argparse.ArgumentParser(
        description="Install latest version of a GitHub repository."
    )
    parser.add_argument(
        "repository",
        help="GitHub repository in the format username/repository",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    latest_tag = get_latest_release_tag(args.repository)
    if latest_tag:
        logging.info(f"Installing {args.repository} at tag {latest_tag}")
        result = install_package(args.repository, latest_tag)
        if result == 0:
            logging.info("Installation successful")
        else:
            logging.error("Installation failed")
    else:
        logging.error("No tags found for the repository.")


if __name__ == "__main__":
    main()

# EOF
