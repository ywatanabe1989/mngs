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

# import docx
# import PyPDF2
# from io import StringIO
# import markdown
# import openpyxl


def load_text(lpath, splitlines=False):
    """
    Load text from a file.

    Parameters:
    lpath (str): The path to the text file.
    splitlines (bool): If True, return a list of lines. If False, return the entire text as a string.

    Returns:
    str or list: The contents of the text file.
    """
    try:
        with open(lpath, "r", encoding="utf-8") as f:
            if splitlines:
                return f.read().splitlines()
            else:
                return f.read()
    except UnicodeDecodeError:
        # If UTF-8 fails, try to detect the encoding
        import chardet

        with open(lpath, "rb") as f:
            raw_data = f.read()
        detected = chardet.detect(raw_data)
        encoding = detected["encoding"]

        with open(lpath, "r", encoding=encoding) as f:
            if splitlines:
                return f.read().splitlines()
            else:
                return f.read()


def load(lpath, show=False, verbose=False, **kwargs):
    if lpath.startswith('f"'):
        lpath = eval(lpath)

    lpath = lpath.replace("/./", "/")
    try:
        extension = "." + lpath.split(".")[-1]

        # csv
        if extension == ".csv":
            index_col = kwargs.get("index_col", 0)
            obj = pd.read_csv(lpath, **kwargs)
            obj = obj.loc[:, ~obj.columns.str.contains("^Unnamed")]
        # tsv
        elif extension == ".tsv":
            obj = pd.read_csv(lpath, sep="\t", **kwargs)
        # excel
        elif extension in [".xls", ".xlsx", ".xlsm", ".xlsb"]:
            obj = pd.read_excel(lpath, **kwargs)
        # parquet
        elif extension == ".parquet":
            obj = pd.read_parquet(lpath, **kwargs)

        # numpy
        elif extension == ".npy":
            obj = np.load(lpath, allow_pickle=True, **kwargs)

        # numpy npz
        elif extension == ".npz":
            obj = np.load(lpath)
            obj = dict(obj)
            obj = [v for v in obj.values()]

        # pkl
        elif extension == ".pkl":
            with open(lpath, "rb") as l:
                obj = pickle.load(l, **kwargs)
        # joblib
        elif extension == ".joblib":
            with open(lpath, "rb") as l:
                obj = joblib.load(l, **kwargs)
        # hdf5
        elif extension == ".hdf5":
            obj = {}
            with h5py.File(lpath, "r") as hf:
                for name in hf:
                    obj[name] = hf[name][:]
        # json
        elif extension == ".json":
            with open(lpath, "r") as f:
                obj = json.load(f)

        # Image
        elif extension in [".jpg", ".png", ".tiff", "tif"]:
            obj = Image.open(lpath)

        # yaml
        elif extension in [".yaml", ".yml"]:

            lower = kwargs.pop("lower", False)

            with open(lpath) as f:
                obj = yaml.safe_load(f, **kwargs)

            if lower:
                obj = {k.lower(): v for k, v in obj.items()}

        # txt
        elif extension in [".txt", ".log", ".event"]:
            with open(lpath, "r") as f:
                obj = f.read().splitlines()

        # md
        elif extension == ".md":
            obj = load_markdown(lpath, **kwargs)

        # pth
        elif extension in [".pth", ".pt"]:
            obj = torch.load(lpath, **kwargs)

        # mat
        elif extension == ".mat":
            from pymatreader import read_mat

            obj = read_mat(lpath, **kwargs)
        # xml
        elif extension == ".xml":
            from xml2dict import xml2dict

            obj = xml2dict(lpath, **kwargs)
        # # edf
        # elif extension == ".edf":  # [REVISED]
        #     obj = mne.io.read_raw_edf(lpath, preload=True)  # [REVISED]
        # con
        elif extension == ".con":
            obj = mne.io.read_raw_fif(lpath, preload=True, **kwargs)
            obj = obj.to_data_frame()
            obj["samp_rate"] = obj.info["sfreq"]
        # # mrk
        # elif extension == ".mrk":  # [REVISED]
        #     obj = mne.io.read_mrk(lpath)  # [REVISED]

        # catboost model
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

        elif extension in TEXT_EXTENSIONS:
            obj = _load_text(lpath)

        # Word document
        elif extension in [".docx", ".doc"]:
            if extension == ".doc":
                import win32com.client

                word = win32com.client.Dispatch("Word.Application")
                doc = word.Documents.Open(lpath)
                obj = [paragraph.Range.Text for paragraph in doc.Paragraphs]
                doc.Close()
                word.Quit()
            else:
                doc = Document(lpath)
                obj = [paragraph.text for paragraph in doc.paragraphs]

        # Excel workbook
        elif extension in [".xlsx", ".xls"]:
            if extension == ".xls":
                import xlrd

                wb = xlrd.open_workbook(lpath)
                obj = {
                    sheet.name: [
                        sheet.row_values(row) for row in range(sheet.nrows)
                    ]
                    for sheet in wb.sheets()
                }
            else:
                wb = load_workbook(lpath, read_only=True)
                obj = {
                    sheet.title: list(sheet.values) for sheet in wb.worksheets
                }

        else:
            print(f"\nNot loaded from: {lpath}\n")
            return None

        if show or verbose:
            print(f"\nLoaded from: {lpath}\n")

        return obj

    except Exception as e:
        raise ValueError(f"{lpath} was not loaded:\n{e}")


# def _load_text(lpath, splitlines=False):
#     with open(lpath, "r") as f:
#         return f.read().splitlines() if splitlines else f.read()


def _load_text(lpath):
    with open(lpath, "r") as f:
        return f.read()


def _load_eeg_data(filename, **kwargs):
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


def load_markdown(lpath_md, style="plain_text"):
    import html2text
    import markdown

    # Load Markdown content from a file
    with open(lpath_md, "r") as file:
        markdown_content = file.read()

    # Convert Markdown to HTML
    html_content = markdown.markdown(markdown_content)
    if style == "html":
        return html_content

    elif style == "plain_text":
        text_maker = html2text.HTML2Text()
        text_maker.ignore_links = True
        text_maker.bypass_tables = False
        plain_text = text_maker.handle(html_content)

        return plain_text


# def _check_encoding(file_path):
#     from chardet.universaldetector import UniversalDetector

#     detector = UniversalDetector()
#     with open(file_path, mode="rb") as f:
#         for binary in f:
#             detector.feed(binary)
#             if detector.done:
#                 break
#     detector.close()
#     enc = detector.result["encoding"]
#     return enc


# def get_data_path_from_a_package(package_str, resource):
#     import importlib
#     import os
#     import sys

#     spec = importlib.util.find_spec(package_str)
#     data_dir = os.path.join(spec.origin.split("src")[0], "data")
#     resource_path = os.path.join(data_dir, resource)
#     return resource_path


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


def load_configs(IS_DEBUG=None, show=False, verbose=False):

    if os.getenv("CI") == "True":
        IS_DEBUG = True

    def update_debug(config, IS_DEBUG):
        if IS_DEBUG:
            debug_keys = mngs.gen.search("^DEBUG_", list(config.keys()))[1]
            for dk in debug_keys:
                dk_wo_debug_prefix = dk.split("DEBUG_")[1]
                config[dk_wo_debug_prefix] = config[dk]
                if show or verbose:
                    print(f"\n{dk} -> {dk_wo_debug_prefix}\n")
        return config

    # Check ./config/IS_DEBUG.yaml file if IS_DEBUG argument is not passed
    if IS_DEBUG is None:
        IS_DEBUG_PATH = "./config/IS_DEBUG.yaml"
        if os.path.exists(IS_DEBUG_PATH):
            IS_DEBUG = mngs.io.load("./config/IS_DEBUG.yaml").get("IS_DEBUG")
        else:
            IS_DEBUG = False

    # Main
    CONFIGS = {}
    for lpath in glob("./config/*.yaml"):
        CONFIG = update_debug(mngs.io.load(lpath), IS_DEBUG)
        CONFIGS.update(CONFIG)

    CONFIGS = mngs.gen.DotDict(CONFIGS)

    return CONFIGS


################################################################################
# dev
################################################################################
# def _load_docx(lpath):
#     doc = docx.Document(lpath)
#     full_text = []
#     for para in doc.paragraphs:
#         full_text.append(para.text)
#     return "\n".join(full_text)


# def _load_pdf(lpath):
#     reader = PyPDF2.PdfReader(lpath)
#     full_text = []
#     for page_num in range(len(reader.pages)):
#         page = reader.pages[page_num]
#         full_text.append(page.extract_text())
#     return "\n".join(full_text)


# def _load_latex(lpath):
#     return lpath.read().decode("utf-8")


# def _load_excel(lpath):
#     workbook = openpyxl.load_workbook(lpath)
#     all_text = []
#     for sheet in workbook:
#         for row in sheet.iter_rows(values_only=True):
#             all_text.append(
#                 " ".join(
#                     [str(cell) if cell is not None else "" for cell in row]
#                 )
#             )
#     return "\n".join(all_text)


# def _load_markdown(lpath):
#     md_text = StringIO(lpath.read().decode("utf-8"))
#     html = markdown.markdown(md_text.read())
#     return html


# def _load_textfile(lpath):
#     return lpath.read().decode("utf-8")
