#!/usr/bin/env python3

import json
import pickle
import warnings

import h5py
import mne
import mngs
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
    """
    Load data from a file with various extensions into an appropriate Python object.

    Arguments:
        lpath (str): Path to the file to be loaded.
        show (bool, optional): If True, prints the path of the loaded file. Defaults to False.
        **kwargs: Additional keyword arguments to pass to the underlying loading functions.

    Returns:
        object: The loaded data as a Python object (e.g., DataFrame, NumPy array, etc.),
                or None if the file type is not supported or the file cannot be loaded.

    Supported file types and their corresponding return types:
        - `.cbm`: CatBoost model (requires CatBoost library)
        - `.con`: pandas.DataFrame with an additional 'samp_rate' key
        - `.csv`: pandas.DataFrame
        - `.edf`: mne.io.Raw (MNE-Python Raw object)
        - `.hdf5`: dict of numpy.ndarrays
        - `.joblib`: object loaded via joblib
        - `.json`: dict or list, depending on the JSON structure
        - `.log`: list of str
        - `.mat`: dict (loaded using pymatreader)
        - `.mrk`: mne.io.kit.mrk (MNE-Python MRK object)
        - `.npy`: numpy.ndarray
        - `.pkl`: object loaded via pickle
        - `.pth`, `.pt`: torch.nn.Module (state dict) or tensor
        - `.tsv`: pandas.DataFrame
        - `.txt`: list of str
        - `.xls`, `.xlsx`, `.xlsm`, `.xlsb`: pandas.DataFrame
        - `.xml`: dict (requires xml2dict implementation)
        - `.yaml`: dict

    Note:
        - The function does not handle image files (.png, .tiff, .tif) and will return None for these types.
        - The function assumes that the xml2dict module is available for XML files.
        - For .pth and .pt files, the function loads the entire object, not just the state dict.
        - For .cbm files, the CatBoost library must be installed, and the object must be a CatBoost model.
    """

    # csv
    if lpath.endswith(".csv"):
        obj = pd.read_csv(lpath, **kwargs)
        unnamed_cols = mngs.gen.search("Unnamed", obj.columns)[1]
        for unnamed_col in unnamed_cols:
            del obj[unnamed_col]
    # tsv
    if lpath.endswith(".tsv"):
        obj = pd.read_csv(lpath, sep="\t", **kwargs)  # [REVISED]

    # excel
    if (
        lpath.endswith(".xls")
        or lpath.endswith(".xlsx")
        or lpath.endswith(".xlsm")
        or lpath.endswith(".xlsb")
    ):
        obj = pd.read_excel(lpath, **kwargs)
    # numpy
    if lpath.endswith(".npy"):
        obj = np.load(lpath)
    # pkl
    if lpath.endswith(".pkl"):
        with open(lpath, "rb") as l:
            obj = pickle.load(l)
    # joblib
    if lpath.endswith(".joblib"):
        with open(lpath, "rb") as l:
            obj = joblib.load(l)
    # hdf5
    if lpath.endswith(".hdf5"):
        obj = {}
        with h5py.File(fpath, "r") as hf:
            for name in name_list:
                obj_tmp = hf[name][:]
                obj[name] = obj_tmp
    # json
    elif lpath.endswith(".json"):
        with open(lpath, "r") as f:
            obj = json.load(f)
    # png
    if lpath.endswith(".png"):
        pass
    # tiff
    if lpath.endswith(".tiff") or lpath.endswith(".tif"):
        pass
    # yaml
    if lpath.endswith(".yaml"):
        obj = {}
        with open(lpath) as f:
            obj_tmp = yaml.safe_load(f)
            obj.update(obj_tmp)
    # txt
    if (lpath.endswith(".txt")) or (lpath.endswith(".log")):
        f = open(lpath, "r")
        obj = [l.strip("\n\r") for l in f]
        f.close()
    # pth
    if lpath.endswith(".pth") or lpath.endswith(".pt"):
        # return model.load_state_dict(torch.load(lpath))
        obj = torch.load(lpath)

    # mat
    if lpath.endswith(".mat"):
        import pymatreader

        obj = pymatreader.read_mat(lpath)
    # xml
    if lpath.endswith("xml"):
        from ._xml2dict import xml2dict

        obj = xml2dict(lpath)
    # edf
    if lpath.endswith("edf"):
        obj = mne.io.read_raw_edf(lpath)
    # con
    if lpath.endswith("con"):
        _obj = mne.io.read_raw(lpath)
        obj = _obj.to_data_frame()
        obj["samp_rate"] = _obj.info.get("sfreq")
    # mrk
    if lpath.endswith("mrk"):
        obj = mne.io.kit.read_mrk(lpath)

    # catboost model
    if lpath.endswith(".cbm"):
        obj = obj.load_model(lpath)

    # if mngs.general.is_defined_local("obj"):
    if "obj" in locals():
        if show:
            print("\nLoaded from: {}\n".format(lpath))
        return obj
    else:
        return None


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
