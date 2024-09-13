#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-13 16:39:20 (ywatanabe)"
# /home/ywatanabe/proj/_mngs_repo_openhands/src/mngs/general/utils/_weather.py

"""This script does XYZ."""

"""Imports"""
import importlib
import logging
import os
import re
import sys
import warnings
from glob import glob
from pprint import pprint

import matplotlib
import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from icecream import ic
from natsort import natsorted
from tqdm import tqdm

try:
    from scripts import utils
except:
    pass

"""Config"""
# CONFIG = mngs.gen.load_configs()

"""Functions & Classes"""
import time

import requests
from plyer import notification

API_KEY = "your_api_key"
CITY = "your_city"
API_URL = (
    f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}"
)


def check_rain():
    response = requests.get(API_URL)
    data = response.json()
    if "rain" in data:
        notify_rain()


def notify_rain():
    notification.notify(
        title="Rain Alert",
        message="It's starting to rain in your area!",
        timeout=10,
    )


while True:
    check_rain()
    time.sleep(300)  # Check every 5 minutes

if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, agg=True
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
