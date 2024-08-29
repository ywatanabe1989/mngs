#!/usr/bin/env python3

import json
import os
import pickle
import warnings
from glob import glob

import h5py
import joblib
import mne
import mngs
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from docx import Document
from openpyxl import load_workbook

def load(lpath, show=False, verbose=False, **kwargs):
    """
    Load data from various file formats.

    This function supports loading data from multiple file formats including CSV, Excel, Numpy, Pickle, JSON, YAML, and more.

    Parameters:
    -----------
    lpath : str
        The path to the file to be loaded.
    show : bool, optional
        If True, display additional information during loading. Default is False.
    verbose : bool, optional
        If True, print verbose output during loading. Default is False.
    **kwargs : dict
        Additional keyword arguments to be passed to the specific loading function.

    Returns:
    --------
    object
        The loaded data object, which can be of various types depending on the input file format.

    Raises:
    -------
    ValueError
        If the file extension is not supported.

    Examples:
    ---------
    >>> data = load('data.csv')
    >>> image = load('image.png')
    >>> model = load('model.pth')
    """
    if lpath.startswith('f"'):
        lpath = eval(lpath)

    lpath = lpath.replace("/./", "/")
    try:
        extension = "." + lpath.split(".")[-1]

        # CSV
        if extension == ".csv":
            index_col = kwargs.get("index_col", 0)
            obj = pd.read_csv(lpath, **kwargs)
            obj = obj.loc[:, ~obj.columns.str.contains("^Unnamed")]
        # TSV
        elif extension == ".tsv":
            obj = pd.read_csv(lpath, sep="\t", **kwargs)
        # Excel
        elif extension in [".xls", ".xlsx", ".xlsm", ".xlsb"]:
            obj = pd.read_excel(lpath, **kwargs)
        # Parquet
        elif extension == ".parquet":
            obj = pd.read_parquet(lpath, **kwargs)
        # Numpy
        elif extension == ".npy":
            obj = np.load(lpath, allow_pickle=True, **kwargs)
        # Numpy NPZ
        elif extension == ".npz":
            obj = np.load(lpath)
            obj = dict(obj)
            obj = [v for v in obj.values()]
        # Pickle
        elif extension == ".pkl":
            with open(lpath, "rb") as l:
                obj = pickle.load(l, **kwargs)
        # Joblib
        elif extension == ".joblib":
            with open(lpath, "rb") as l:
                obj = joblib.load(l, **kwargs)
        # HDF5
        elif extension == ".hdf5":
            obj = {}
            with h5py.File(lpath, "r") as hf:
                for name in hf:
                    obj[name] = hf[name][:]
        # JSON
        elif extension == ".json":
            with open(lpath, "r") as f:
                obj = json.load(f)
        # Image
        elif extension in [".jpg", ".png", ".tiff", "tif"]:
            obj = Image.open(lpath)
        # YAML
        elif extension in [".yaml", ".yml"]:
            lower = kwargs.pop("lower", False)
            with open(lpath) as f:
                obj = yaml.safe_load(f, **kwargs)
            if lower:
                obj = {k.lower(): v for k, v in obj.items()}
        # Text
        elif extension in [".txt", ".log", ".event"]:
            with open(lpath, "r") as f:
                obj = f.read().splitlines()
        # Markdown
        elif extension == ".md":
            obj = load_markdown(lpath, **kwargs)
        # PyTorch
        elif extension in [".pth", ".pt"]:
            obj = torch.load(lpath, **kwargs)
        # MATLAB
        elif extension == ".mat":
            from pymatreader import read_mat
            obj = read_mat(lpath, **kwargs)
        # XML
        elif extension == ".xml":
            from xml2dict import xml2dict
            obj = xml2dict(lpath, **kwargs)
        # CON
        elif extension == ".con":
            obj = mne.io.read_raw_fif(lpath, preload=True, **kwargs)
            obj = obj.to_data_frame()
            obj["samp_rate"] = obj.info["sfreq"]
        # CatBoost model
        elif extension == ".cbm":
            from catboost import CatBoostModel
            obj = CatBoostModel.load_model(lpath, **kwargs)
        # EEG data
        elif extension in [
            ".vhdr",
            ".vmrk",
            ".edf",
            ".bdf",
            ".gdf",
            ".cnt",
            ".egi",
            ".eeg",
            ".set",
        ]:
            obj = _load_eeg_data(lpath, **kwargs)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")

        return obj
    except Exception as e:
        print(f"Error loading file {lpath}: {str(e)}")
        raise

# ... (rest of the file remains unchanged)

