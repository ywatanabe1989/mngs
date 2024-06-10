#!/usr/bin/env python3

import csv
import inspect
import io as _io
import json
import os
import pickle
from shutil import move

import h5py
import joblib
import mngs
import numpy as np
import pandas as pd
import plotly
import scipy
import torch
from matplotlib import animation
from PIL import Image
from ruamel.yaml import YAML


def save(
    obj,
    sfname_or_spath,
    makedirs=True,
    verbose=True,
    from_cwd=False,
    dry_run=False,
    **kwargs,
):
    """
    Saves an object to a file with the specified format, determined by the file extension.
    The function supports saving data in various formats including CSV, NPY, PKL, JOBLIB, PNG, HTML, TIFF, MP4, YAML, JSON, HDF5, PTH, MAT, and CBM.

    Arguments:
        obj (object): The object to be saved. Can be a NumPy array, PyTorch tensor, Pandas DataFrame, or any serializable object depending on the file format.
        sfname_or_spath (str): The file name or path where the object should be saved. The file extension determines the format.
        makedirs (bool, optional): If True, the function will create the directory path if it does not exist. Defaults to True.
        verbose (bool, optional): If True, prints a message upon successful saving. Defaults to True.
        **kwargs: Additional keyword arguments to pass to the underlying save function of the specific format.

    Returns:
        None. The function saves the object to a file and does not return any value.

    Supported Formats:
        - .csv: Pandas DataFrames or listed scalars.
        - .npy: NumPy arrays.
        - .npz: NumPy arrays.
        - .pkl: Serializable Python objects using pickle.
        - .joblib: Objects using joblib with compression.
        - .png: Images, including Plotly figures and Matplotlib plots.
        - .tiff / .tif: Images in TIFF format, typically from Matplotlib plots.
        - .jpeg / .jpg: Images in JPEG format, typically from Matplotlib plots.
        - .svg : Images in the SVG format, typically from Matplotlib plots.
        - .html: Plotly figures as HTML files.
        - .mp4: Videos, likely from animations created with Matplotlib.
        - .yaml: Data in YAML format.
        - .json: Data in JSON format.
        - .hdf5: Data in HDF5 format, useful for large datasets.
        - .pth: PyTorch model states.
        - .mat: Data in MATLAB format using `scipy.io.savemat`.
        - .cbm: CatBoost models.

    Note:
        - The function dynamically selects the appropriate saving mechanism based on the file extension.
        - Ensure that the object type is compatible with the chosen file format.
        - For custom saving mechanisms or unsupported formats, consider extending the function or using the specific library's save function directly.

    Example:
        import mngs

        import numpy as np
        import pandas as pd
        import torch
        import matplotlib.pyplot as plt

        # .npy
        arr = np.array([1, 2, 3])
        mngs.io.save(arr, "xxx.npy")
        # arr = mngs.io.load("xxx.npy")

        # .csv
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        mngs.io.save(df, "xxx.csv")
        # df = mngs.io.load("xxx.csv")

        # .pth
        tensor = torch.tensor([1, 2, 3])
        mngs.io.save(obj, "xxx.pth")
        # tensor = mngs.io.load("xxx.pth")

        # .pkl
        _dict = {"a": 1, "b": 2, "c": [3, 4, 5]} # serializable object like dict, list, ...
        mngs.io.save(_dict, "xxx.pkl")
        # _dict = mngs.io.load("xxx.pkl")

        # .png | .tiff | .jpg
        plt.figure()
        plt.plot(np.array([1, 2, 3]))
        mngs.io.save(plt, "xxx.png") # "xxx.tiff", "xxx.tif", "xxx.jpeg, or "xxx.jpg"

        # .yaml
        mngs.io.save(_dict, "xxx.yaml")
        # _dict = mngs.io.load("xxx.yaml")

        # .json
        mngs.io.save(_dict, "xxx.json")
        # _dict = mngs.io.load("xxx.json")
    """
    # import inspect
    # import json
    # import os
    # import pickle

    # import h5py
    # import numpy as np
    # import pandas as pd
    # import plotly
    # import torch
    # import yaml

    ########################################
    # Determines the save directory from the script.
    # This process should be in this function for the intended behavior of inspect.
    spath, sfname = None, None

    if sfname_or_spath.startswith("/"):
        spath = sfname_or_spath

    # elif sfname_or_spath.startswith("./"):
    else:
        if from_cwd:
            spath_cwd = os.getcwd() + "/" + sfname_or_spath
            spath_cwd = spath_cwd.replace("/./", "/").replace("//", "/")

        fpath = inspect.stack()[1].filename

        if "ipython" in fpath:
            fpath = f'/tmp/fake-{os.getenv("USER")}.py'

        fdir, fname, _ = mngs.path.split(fpath)
        spath = fdir + fname + "/" + sfname_or_spath

    # print(sfname_or_spath, fpath, spath, sfname)
    # Corrects the spath
    spath = spath.replace("/./", "/").replace("//", "/")
    ########################################

    if dry_run:
        print(mngs.gen.ct(f"\n(dry run) Saved to: {spath}", c="yellow"))
        return

    # Makes directory
    if makedirs:
        os.makedirs(os.path.dirname(spath), exist_ok=True)

    _save(obj, spath, verbose=verbose, **kwargs)

    if from_cwd:
        if spath != spath_cwd:
            mngs.sh(f"ln -sf {spath} {spath_cwd}", verbose=False)
            print(
                mngs.gen.color_text(f"\n(Symlinked to: {spath_cwd})", "yellow")
            )


def _save(obj, spath, verbose=True, **kwargs):
    # Main
    try:
        ## copy files
        is_copying_files = (
            isinstance(obj, str) or mngs.gen.is_listed_X(obj, str)
        ) and (isinstance(spath, str) or mngs.gen.is_listed_X(spath, str))
        if is_copying_files:
            mngs.general.copy_files(obj, spath)

        # csv
        elif spath.endswith(".csv"):
            if isinstance(obj, pd.Series):  # Series
                obj.to_csv(spath, **kwargs)
            if isinstance(obj, pd.DataFrame):  # DataFrame
                obj.to_csv(spath, **kwargs)
            if mngs.gen.is_listed_X(obj, [int, float]):  # listed scalars
                _save_listed_scalars_as_csv(
                    obj,
                    spath,
                    **kwargs,
                )
            if mngs.gen.is_listed_X(obj, pd.DataFrame):  # listed DataFrame
                _save_listed_dfs_as_csv(obj, spath, **kwargs)

        # numpy
        elif spath.endswith(".npy"):
            np.save(spath, obj)

        # numpy npz
        elif spath.endswith(".npz"):
            if isinstance(obj, dict):
                np.savez_compressed(spath, **obj)
            elif isinstance(obj, (list, tuple)) and all(
                isinstance(x, np.ndarray) for x in obj
            ):
                obj = {str(ii): obj[ii] for ii in range(len(obj))}
                np.savez_compressed(spath, **obj)
            else:
                raise ValueError(
                    "For .npz files, obj must be a dict of arrays or a list/tuple of arrays."
                )
        # pkl
        elif spath.endswith(".pkl"):
            with open(spath, "wb") as s:  # 'w'
                pickle.dump(obj, s)
        # joblib
        elif spath.endswith(".joblib"):
            with open(spath, "wb") as s:  # 'w'
                joblib.dump(obj, s, compress=3)

        # png
        elif spath.endswith(".png"):
            # plotly
            if isinstance(obj, plotly.graph_objs.Figure):
                obj.write_image(file=spath, format="png")
            # PIL image
            elif isinstance(obj, Image.Image):
                obj.save(spath)
            # matplotlib
            else:
                try:
                    obj.savefig(spath)
                except:
                    obj.figure.savefig(spath)
            del obj

        # html
        elif spath.endswith(".html"):
            # plotly
            if isinstance(obj, plotly.graph_objs.Figure):
                obj.write_html(file=spath)

        # tiff
        elif spath.endswith(".tiff") or spath.endswith(".tif"):
            # PIL image
            if isinstance(obj, Image.Image):
                obj.save(spath)
            # matplotlib
            else:
                try:
                    obj.savefig(spath, dpi=300, format="tiff")
                except:
                    obj.figure.savefig(spath, dpi=300, format="tiff")

            del obj

        # jpeg
        elif spath.endswith(".jpeg") or spath.endswith(".jpg"):
            buf = _io.BytesIO()

            # plotly
            if isinstance(obj, plotly.graph_objs.Figure):
                obj.write_image(
                    buf, format="png"
                )  # Saving plotly figure to buffer as PNG
                buf.seek(0)
                img = Image.open(buf)
                img.convert("RGB").save(spath, "JPEG")
                buf.close()
            # PIL image
            elif isinstance(obj, Image.Image):
                obj.save(spath)
            # matplotlib
            else:
                try:
                    obj.savefig(buf, format="png")
                except:
                    obj.figure.savefig(buf, format="png")

                buf.seek(0)
                img = Image.open(buf)
                img.convert("RGB").save(
                    spath, "JPEG"
                )  # Convert to JPEG and save
                buf.close()
            del obj

        # SVG
        elif spath.endswith(".svg"):
            # Plotly
            if isinstance(obj, plotly.graph_objs.Figure):
                obj.write_image(file=spath, format="svg")
            # Matplotlib
            else:
                try:
                    obj.savefig(spath, format="svg")
                except AttributeError:
                    obj.figure.savefig(spath, format="svg")
            del obj

        # mp4
        elif spath.endswith(".mp4"):
            obj.save(
                spath, writer="ffmpeg", **kwargs
            )  # obj is matplotlib.animation.FuncAnimation
            del obj
            # _mk_mp4(obj, spath)  # obj is matplotlib.pyplot.figure object
            # del obj

        # yaml
        elif spath.endswith(".yaml"):
            yaml = YAML()
            yaml.preserve_quotes = (
                True  # Optional: if you want to preserve quotes
            )
            yaml.indent(
                mapping=4, sequence=4, offset=4
            )  # Optional: set indentation

            with open(spath, "w") as f:
                yaml.dump(obj, f)

        # json
        elif spath.endswith(".json"):
            with open(spath, "w") as f:
                json.dump(obj, f, indent=4)

        # hdf5
        elif spath.endswith(".hdf5"):
            name_list, obj_list = []
            for k, v in obj.items():
                name_list.append(k)
                obj_list.append(v)
            with h5py.File(spath, "w") as hf:
                for (name, obj) in zip(name_list, obj_list):
                    hf.create_dataset(name, data=obj)
        # pth
        elif spath.endswith(".pth"):
            torch.save(obj, spath)

        # mat
        elif spath.endswith(".mat"):
            scipy.io.savemat(spath, obj)

        # catboost model
        elif spath.endswith(".cbm"):
            obj.save_model(spath)

        else:
            raise ValueError("obj was not saved.")

    except Exception as e:
        print(e)

    else:
        if verbose and not is_copying_files:
            file_size = mngs.path.file_size(spath)
            print(
                mngs.gen.ct(f"\nSaved to: {spath} ({file_size})", c="yellow")
            )


# def check_encoding(file_path):
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


def _save_listed_scalars_as_csv(
    listed_scalars,
    spath_csv,
    column_name="_",
    indi_suffix=None,
    round=3,
    overwrite=False,
    verbose=False,
):
    """Puts to df and save it as csv"""

    if overwrite == True:
        _mv_to_tmp(spath_csv, L=2)
    indi_suffix = (
        np.arange(len(listed_scalars)) if indi_suffix is None else indi_suffix
    )
    df = pd.DataFrame(
        {"{}".format(column_name): listed_scalars}, index=indi_suffix
    ).round(round)
    df.to_csv(spath_csv)
    if verbose:
        print("\nSaved to: {}\n".format(spath_csv))


def _save_listed_dfs_as_csv(
    listed_dfs,
    spath_csv,
    indi_suffix=None,
    overwrite=False,
    verbose=False,
):
    """listed_dfs:
        [df1, df2, df3, ..., dfN]. They will be written vertically in the order.

    spath_csv:
        /hoge/fuga/foo.csv

    indi_suffix:
        At the left top cell on the output csv file, '{}'.format(indi_suffix[i])
        will be added, where i is the index of the df.On the other hand,
        when indi_suffix=None is passed, only '{}'.format(i) will be added.
    """
    import numpy as np

    if overwrite == True:
        _mv_to_tmp(spath_csv, L=2)

    indi_suffix = (
        np.arange(len(listed_dfs)) if indi_suffix is None else indi_suffix
    )
    for i, df in enumerate(listed_dfs):
        with open(spath_csv, mode="a") as f:
            f_writer = csv.writer(f)
            i_suffix = indi_suffix[i]
            f_writer.writerow(["{}".format(indi_suffix[i])])
        df.to_csv(spath_csv, mode="a", index=True, header=True)
        with open(spath_csv, mode="a") as f:
            f_writer = csv.writer(f)
            f_writer.writerow([""])
    if verbose:
        print("Saved to: {}".format(spath_csv))


def _mv_to_tmp(fpath, L=2):
    try:
        tgt_fname = mngs.gen.connect_strs(fpath.split("/")[-L:], filler="-")
        tgt_fpath = "/tmp/{}".format(tgt_fname)
        move(fpath, tgt_fpath)
        print("Moved to: {}".format(tgt_fpath))
    except:
        pass


def _mk_mp4(fig, spath_mp4):
    axes = fig.get_axes()

    def init():
        return (fig,)

    def animate(i):
        for ax in axes:
            ax.view_init(elev=10.0, azim=i)
        return (fig,)

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=360, interval=20, blit=True
    )

    writermp4 = animation.FFMpegWriter(
        fps=60, extra_args=["-vcodec", "libx264"]
    )
    anim.save(spath_mp4, writer=writermp4)
    print("\nSaving to: {}\n".format(spath_mp4))


def save_optuna_study_as_csv_and_pngs(study, sdir):
    import optuna

    ## Trials DataFrame
    trials_df = study.trials_dataframe()

    ## Figures
    hparams_keys = list(study.best_params.keys())
    slice_plot = optuna.visualization.plot_slice(study, params=hparams_keys)
    contour_plot = optuna.visualization.plot_contour(
        study, params=hparams_keys
    )
    optim_hist_plot = optuna.visualization.plot_optimization_history(study)
    parallel_coord_plot = optuna.visualization.plot_parallel_coordinate(
        study, params=hparams_keys
    )
    hparam_importances_plot = optuna.visualization.plot_param_importances(
        study
    )
    figs_dict = dict(
        slice_plot=slice_plot,
        contour_plot=contour_plot,
        optim_hist_plot=optim_hist_plot,
        parallel_coord_plot=parallel_coord_plot,
        hparam_importances_plot=hparam_importances_plot,
    )

    ## Saves
    save(trials_df, sdir + "trials_df.csv")

    for figname, fig in figs_dict.items():
        save(fig, sdir + f"{figname}.png")
