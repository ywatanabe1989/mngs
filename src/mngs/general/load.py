#!/usr/bin/env python3

import json
import os
import pickle
import warnings

import h5py
import joblib  # [REVISED]
import mne
import numpy as np
import pandas as pd
import torch
import yaml

if "general" in __file__:
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn(
            '\n"mngs.general.load" will be removed. '
            'Please use "mngs.io.load" instead.',
            PendingDeprecationWarning,
        )


def load(lpath, show=False, **kwargs):
    extension = "." + lpath.split(".")[-1]  # [REVISED]

    # csv
    if extension == ".csv":
        obj = pd.read_csv(lpath, **kwargs)
        obj = obj.loc[:, ~obj.columns.str.contains("^Unnamed")]  # [REVISED]
    # tsv
    elif extension == ".tsv":
        obj = pd.read_csv(lpath, sep="\t", **kwargs)

    # excel
    elif extension in [".xls", ".xlsx", ".xlsm", ".xlsb"]:  # [REVISED]
        obj = pd.read_excel(lpath, **kwargs)

    # numpy
    elif extension == ".npy":
        obj = np.load(lpath, allow_pickle=True)  # [REVISED]
    # pkl
    elif extension == ".pkl":
        with open(lpath, "rb") as l:
            obj = pickle.load(l)
    # joblib
    elif extension == ".joblib":
        with open(lpath, "rb") as l:
            obj = joblib.load(l)
    # hdf5
    elif extension == ".hdf5":
        obj = {}
        with h5py.File(lpath, "r") as hf:
            for name in hf:  # [REVISED]
                obj[name] = hf[name][:]
    # json
    elif extension == ".json":
        with open(lpath, "r") as f:
            obj = json.load(f)
    # png
    elif extension == ".png":
        pass
    # tiff
    elif extension in [".tiff", ".tif"]:
        pass
    # yaml
    elif extension == ".yaml":
        with open(lpath) as f:
            obj = yaml.safe_load(f)  # [REVISED]
    # txt
    elif extension in [".txt", ".log"]:
        with open(lpath, "r") as f:  # [REVISED]
            obj = f.read().splitlines()  # [REVISED]
    # pth
    elif extension in [".pth", ".pt"]:
        obj = torch.load(lpath)

    # mat
    elif extension == ".mat":  # [REVISED]
        from pymatreader import read_mat  # [REVISED]

        obj = read_mat(lpath)  # [REVISED]
    # xml
    elif extension == ".xml":  # [REVISED]
        from xml2dict import xml2dict  # [REVISED]

        obj = xml2dict(lpath)  # [REVISED]
    # # edf
    # elif extension == ".edf":  # [REVISED]
    #     obj = mne.io.read_raw_edf(lpath, preload=True)  # [REVISED]
    # con
    elif extension == ".con":  # [REVISED]
        obj = mne.io.read_raw_fif(lpath, preload=True)  # [REVISED]
        obj = obj.to_data_frame()  # [REVISED]
        obj["samp_rate"] = obj.info["sfreq"]  # [REVISED]
    # # mrk
    # elif extension == ".mrk":  # [REVISED]
    #     obj = mne.io.read_mrk(lpath)  # [REVISED]

    # catboost model
    elif extension == ".cbm":  # [REVISED]
        from catboost import CatBoostModel  # [REVISED]

        obj = CatBoostModel.load_model(lpath)  # [REVISED]

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
        obj = load_eeg_data(lpath, **kwargs)

    else:
        print(f"\nNot loaded from: {lpath}\n")
        return None

    if show:
        print(f"\nLoaded from: {lpath}\n")

    return obj


def load_eeg_data(filename, **kwargs):
    """
    Load EEG data based on file extension and associated files using MNE-Python.

    Parameters:
    filename (str: The path to the file to be loaded.

    Returns:
    raw (mne.io.Raw: The loaded raw EEG data.
    """
    # Get the file extension
    extension = filename.split(".")[-1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        # Load the data based on the file extension
        if extension in ["vhdr", "vmrk"]:
            # Load BrainVision data
            raw = mne.io.read_raw_brainvision(filename, preload=True, **kwargs)
        elif extension == "edf":
            # Load European data format
            raw = mne.io.read_raw_edf(filename, preload=True, **kwargs)
        elif extension == "bdf":
            # Load BioSemi data format
            raw = mne.io.read_raw_bdf(filename, preload=True, **kwargs)
        elif extension == "gdf":
            # Load General data format
            raw = mne.io.read_raw_gdf(filename, preload=True, **kwargs)
        elif extension == "cnt":
            # Load Neuroscan CNT data
            raw = mne.io.read_raw_cnt(filename, preload=True, **kwargs)
        elif extension == "egi":
            # Load EGI simple binary data
            raw = mne.io.read_raw_egi(filename, preload=True, **kwargs)
        elif extension == "set":
            # ???
            raw = mne.io.read_raw(filename, preload=True, **kwargs)
        elif extension == "eeg":
            is_BrainVision = any(
                os.path.isfile(filename.replace(".eeg", ext))
                for ext in [".vhdr", ".vmrk"]
            )
            is_NihonKoden = any(
                os.path.isfile(filename.replace(".eeg", ext))
                for ext in [".21e", ".pnt", ".log"]
            )

            # Brain Vision
            if is_BrainVision:
                filename_v = filename.replace(".eeg", ".vhdr")
                raw = mne.io.read_raw_brainvision(
                    filename_v, preload=True, **kwargs
                )
            # Nihon Koden
            if is_NihonKoden:
                # raw = mne.io.read_raw_nihon(filename, preload=True, **kwargs)
                raw = mne.io.read_raw(filename, preload=True, **kwargs)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")

        return raw


def check_encoding(file_path):
    from chardet.universaldetector import UniversalDetector

    detector = UniversalDetector()
    with open(file_path, mode="rb") as f:
        for binary in f:
            detector.feed(binary)
            if detector.done:
                break
    detector.close()
    enc = detector.result["encoding"]
    return enc


def get_data_path_from_a_package(package_str, resource):
    import importlib
    import os
    import sys

    spec = importlib.util.find_spec(package_str)
    data_dir = os.path.join(spec.origin.split("src")[0], "data")
    resource_path = os.path.join(data_dir, resource)
    return resource_path


def load_yaml_as_an_optuna_dict(fpath_yaml, trial):
    _d = load(fpath_yaml)

    for k, v in _d.items():

        dist = v["distribution"]

        if dist == "categorical":
            _d[k] = trial.suggest_categorical(k, v["values"])

        elif dist == "uniform":
            _d[k] = trial.suggest_int(k, float(v["min"]), float(v["max"]))

        elif dist == "loguniform":
            _d[k] = trial.suggest_loguniform(
                k, float(v["min"]), float(v["max"])
            )

        elif dist == "intloguniform":
            _d[k] = trial.suggest_int(
                k, float(v["min"]), float(v["max"]), log=True
            )

    return _d


def load_study_rdb(study_name, rdb_raw_bytes_url):
    """
    study = load_study_rdb(
        study_name="YOUR_STUDY_NAME",
        rdb_raw_bytes_url="sqlite:///*.db"
    )
    """
    import optuna

    # rdb_raw_bytes_url = "sqlite:////tmp/fake/ywatanabe/_MicroNN_WindowSize-1.0-sec_MaxEpochs_100_2021-1216-1844/optuna_study_test_file#0.db"
    storage = optuna.storages.RDBStorage(url=rdb_raw_bytes_url)
    study = optuna.load_study(study_name=study_name, storage=storage)
    print(f"\nLoaded: {rdb_raw_bytes_url}\n")
    return study
