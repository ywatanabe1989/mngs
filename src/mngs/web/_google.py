#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-29 23:39:18 (ywatanabe)"
# /home/ywatanabe/proj/mngs_repo/src/mngs/web/_google.py


"""
This script does XYZ.
"""


"""
Imports
"""
import os
import re
import sys

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import importlib

import mngs

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
from natsort import natsorted
from glob import glob
from pprint import pprint
import warnings
import logging
from tqdm import tqdm
import xarray as xr

# sys.path = ["."] + sys.path
# from scripts import utils, load

"""
Warnings
"""
# warnings.simplefilter("ignore", UserWarning)


"""
Config
"""
# CONFIG = mngs.gen.load_configs()


"""
Functions & Classes
"""


def main2():
    import requests
    from bs4 import BeautifulSoup

    query = "Python programming"
    url = f"https://www.google.com/search?q={query}"

    # Send a GET request
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # links = soup.find_all("a")
        # # Extract the URLs from href attributes
        # urls = []
        # for link in links:
        #     href = link.get("href")  # Get the href attribute
        #     if href:  # If href is not None
        #         urls.append(href)

        # # Print the extracted URLs
        # for u in urls:
        #     print(u)

        # Find all search result links
        for g in soup.find_all("h3"):
            print(g.get_text())
    else:
        print(f"Failed to retrieve results: {response.status_code}")


def main():
    from googlesearch import search

    # Query
    query = "Python programming"

    # Perform the search
    for result in search(query, num_results=10):
        print(result)
    pass


if __name__ == "__main__":
    # # Argument Parser
    # import argparse
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--var', '-v', type=int, default=1, help='')
    # parser.add_argument('--flag', '-f', action='store_true', default=False, help='')
    # args = parser.parse_args()

    # Main
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )
    main()
    # main2()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
