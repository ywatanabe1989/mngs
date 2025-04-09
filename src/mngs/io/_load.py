#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-10 08:05:53 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/src/mngs/io/_load.py
# ----------------------------------------
import os
__FILE__ = (
    "/ssh:sp:/home/ywatanabe/proj/mngs_repo/src/mngs/io/_load.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

from typing import Any
from ..decorators import preserve_doc
from ..str._clean_path import clean_path
# from ._load_modules._catboost import _load_catboost
from ._load_modules._con import _load_con
from ._load_modules._db import _load_sqlite3db
from ._load_modules._docx import _load_docx
from ._load_modules._eeg import _load_eeg_data
from ._load_modules._hdf5 import _load_hdf5
from ._load_modules._image import _load_image
from ._load_modules._joblib import _load_joblib
from ._load_modules._json import _load_json
from ._load_modules._markdown import _load_markdown
from ._load_modules._numpy import _load_npy
from ._load_modules._matlab import _load_matlab
from ._load_modules._pandas import _load_csv, _load_excel, _load_tsv
from ._load_modules._pdf import _load_pdf
from ._load_modules._pickle import _load_pickle
from ._load_modules._torch import _load_torch
from ._load_modules._txt import _load_txt
from ._load_modules._xml import _load_xml
from ._load_modules._yaml import _load_yaml
from ._load_modules._matlab import _load_matlab

def load(
    lpath: str, show: bool = False, verbose: bool = False, **kwargs
) -> Any:
    """
    Load data from various file formats.

    This function supports loading data from multiple file formats.

    Parameters
    ----------
    lpath : str
        The path to the file to be loaded.
    show : bool, optional
        If True, display additional information during loading. Default is False.
    verbose : bool, optional
        If True, print verbose output during loading. Default is False.
    **kwargs : dict
        Additional keyword arguments to be passed to the specific loading function.

    Returns
    -------
    object
        The loaded data object, which can be of various types depending on the input file format.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    FileNotFoundError
        If the specified file does not exist.

    Supported Extensions
    -------------------
    - Data formats: .csv, .tsv, .xls, .xlsx, .xlsm, .xlsb, .json, .yaml, .yml
    - Scientific: .npy, .npz, .mat, .hdf5, .con
    - ML/DL: .pth, .pt, .cbm, .joblib, .pkl
    - Documents: .txt, .log, .event, .md, .docx, .pdf, .xml
    - Images: .jpg, .png, .tiff, .tif
    - EEG data: .vhdr, .vmrk, .edf, .bdf, .gdf, .cnt, .egi, .eeg, .set
    - Database: .db

    Examples
    --------
    >>> data = load('data.csv')
    >>> image = load('image.png')
    >>> model = load('model.pth')
    """
    lpath = clean_path(lpath)

    if not os.path.exists(lpath):
        raise FileNotFoundError(f"{lpath} not found.")

    loaders_dict = {
        # Default
        "": _load_txt,
        # Config/Settings
        "yaml": _load_yaml,
        "yml": _load_yaml,
        "json": _load_json,
        "xml": _load_xml,
        # ML/DL Models
        "pth": _load_torch,
        "pt": _load_torch,
        # "cbm": _load_catboost,
        "joblib": _load_joblib,
        "pkl": _load_pickle,
        # Tabular Data
        "csv": _load_csv,
        "tsv": _load_tsv,
        "xls": _load_excel,
        "xlsx": _load_excel,
        "xlsm": _load_excel,
        "xlsb": _load_excel,
        "db": _load_sqlite3db,
        # Scientific Data
        "npy": _load_npy,
        "npz": _load_npy,
        "mat": _load_matlab,
        "hdf5": _load_hdf5,
        "mat": _load_matlab,
        "con": _load_con,
        # Documents
        "txt": _load_txt,
        "tex": _load_txt,
        "log": _load_txt,
        "event": _load_txt,
        "py": _load_txt,
        "sh": _load_txt,
        "md": _load_markdown,
        "docx": _load_docx,
        "pdf": _load_pdf,
        # Images
        "jpg": _load_image,
        "png": _load_image,
        "tiff": _load_image,
        "tif": _load_image,
        # EEG Data
        "vhdr": _load_eeg_data,
        "vmrk": _load_eeg_data,
        "edf": _load_eeg_data,
        "bdf": _load_eeg_data,
        "gdf": _load_eeg_data,
        "cnt": _load_eeg_data,
        "egi": _load_eeg_data,
        "eeg": _load_eeg_data,
        "set": _load_eeg_data,
    }

    ext = lpath.split(".")[-1] if "." in lpath else ""
    loader = preserve_doc(loaders_dict.get(ext, _load_txt))

    try:
        return loader(lpath, **kwargs)
    except (ValueError, FileNotFoundError) as e:
        raise ValueError(f"Error loading file {lpath}: {str(e)}")

# EOF