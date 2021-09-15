#!/usr/bin/env python3

import csv

import pandas as pd
import mngs
import numpy as np


def save(obj, sfname_or_spath, makedirs=True, show=True, **kwargs):
    """
    Example
      save(arr, 'data.npy')
      save(df, 'df.csv')
      save(serializable, 'serializable.pkl')
    """
    import inspect
    import os
    import pickle

    import h5py
    import numpy as np
    import pandas as pd
    import torch
    import yaml

    spath, sfname = None, None

    if "/" in sfname_or_spath:
        spath = sfname_or_spath
    else:
        sfname = sfname_or_spath

    if (spath is None) and (sfname is not None):
        ## for ipython
        __file__ = inspect.stack()[1].filename
        if "ipython" in __file__:
            __file__ = "/tmp/fake.py"

        ## spath
        fpath = __file__
        fdir, fname, _ = mngs.general.split_fpath(fpath)
        sdir = fdir + fname + "/"
        spath = sdir + sfname
        # spath = mk_spath(sfname, makedirs=True)

    ## Make directory
    if makedirs:
        sdir = os.path.dirname(spath)
        os.makedirs(sdir, exist_ok=True)

    ## Saves
    # csv
    if spath.endswith(".csv"):
        if isinstance(obj, pd.DataFrame):  # DataFrame
            obj.to_csv(spath)
        if is_listed_X(obj, [int, float]):  # listed scalars
            save_listed_scalars_as_csv(
                obj,
                spath,
                **kwargs,
            )
        if is_listed_X(obj, pd.DataFrame):  # listed DataFrame
            # save_listed_dfs_as_csv(obj, spath, indi_suffix=None, overwrite=False)
            save_listed_dfs_as_csv(obj, spath, **kwargs)
    # numpy
    if spath.endswith(".npy"):
        np.save(spath, obj)
    # pkl
    if spath.endswith(".pkl"):
        with open(spath, "wb") as s:  # 'w'
            pickle.dump(obj, s)
    # joblib
    if spath.endswith(".joblib"):
        with open(spath, "wb") as s:  # 'w'
            joblib.dump(obj, s, compress=3)
    # png
    if spath.endswith(".png"):
        try:
            obj.savefig(spath)
        except:
            obj.figure.savefig(spath)
        del obj
    # tiff
    if spath.endswith(".tiff") or spath.endswith(".tif"):
        obj.savefig(spath, dpi=300, format="tiff")  # obj is matplotlib.pyplot object
        del obj
    # mp4
    if spath.endswith(".mp4"):
        mk_mp4(obj, spath)  # obj is matplotlib.pyplot.figure object
        del obj
    # yaml
    if spath.endswith(".yaml"):
        with open(spath, "w") as f:
            yaml.dump(obj, f)
    # hdf5
    if spath.endswith(".hdf5"):
        name_list, obj_list = []
        for k, v in obj.items():
            name_list.append(k)
            obj_list.append(v)
        with h5py.File(spath, "w") as hf:
            for (name, obj) in zip(name_list, obj_list):
                hf.create_dataset(name, data=obj)
    # pth
    if spath.endswith(".pth"):
        # torch.save(obj.state_dict(), spath)
        torch.save(obj, spath)

    if show:
        print("\nSaved to: {s}\n".format(s=spath))


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


def is_listed_X(obj, types):
    """
    Example:
        obj = [3, 2, 1, 5]
        is_listed_X(obj,
    """
    import numpy as np

    try:
        conditions = []
        condition_list = isinstance(obj, list)

        if not (isinstance(types, list) or isinstance(types, tuple)):
            types = [types]

        _conditions_susp = []
        for typ in types:
            _conditions_susp.append(
                (np.array([isinstance(o, typ) for o in obj]) == True).all()
            )

        condition_susp = np.any(_conditions_susp)

        is_listed_X = np.all([condition_list, condition_susp])
        return is_listed_X

    except:
        return False


def save_listed_scalars_as_csv(
    listed_scalars,
    spath_csv,
    column_name="_",
    indi_suffix=None,
    round=3,
    overwrite=False,
    show=False,
):
    """Puts to df and save it as csv"""
    import numpy as np

    if overwrite == True:
        mv_to_tmp(spath_csv, L=2)
    indi_suffix = np.arange(len(listed_scalars)) if indi_suffix is None else indi_suffix
    df = pd.DataFrame(
        {"{}".format(column_name): listed_scalars}, index=indi_suffix
    ).round(round)
    df.to_csv(spath_csv)
    if show:
        print("\nSaved to: {}\n".format(spath_csv))


def save_listed_dfs_as_csv(
    listed_dfs,
    spath_csv,
    indi_suffix=None,
    overwrite=False,
    show=False,
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
        mv_to_tmp(spath_csv, L=2)

    indi_suffix = np.arange(len(listed_dfs)) if indi_suffix is None else indi_suffix
    for i, df in enumerate(listed_dfs):
        with open(spath_csv, mode="a") as f:
            f_writer = csv.writer(f)
            i_suffix = indi_suffix[i]
            f_writer.writerow(["{}".format(indi_suffix[i])])
        df.to_csv(spath_csv, mode="a", index=True, header=True)
        with open(spath_csv, mode="a") as f:
            f_writer = csv.writer(f)
            f_writer.writerow([""])
    if show:
        print("Saved to: {}".format(spath_csv))


def mk_mp4(fig, spath_mp4):
    from matplotlib import animation

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

    writermp4 = animation.FFMpegWriter(fps=60, extra_args=["-vcodec", "libx264"])
    anim.save(spath_mp4, writer=writermp4)
    print("\nSaving to: {}\n".format(spath_mp4))


def mv_to_tmp(fpath, L=2):
    import os
    from shutil import move

    try:
        tgt_fname = connect_strs_with_hyphens(fpath.split("/")[-L:])
        tgt_fpath = "/tmp/{}".format(tgt_fname)
        move(fpath, tgt_fpath)
        print("Moved to: {}".format(tgt_fpath))
    except:
        pass


# def save(obj, sfname_or_spath, makedirs=True):
#     """
#     Example
#       save(arr, 'data.npy')
#       save(df, 'df.csv')
#       save(serializable, 'serializable.pkl')
#     """
#     import inspect
#     import os
#     import pickle

#     import h5py
#     import numpy as np
#     import pandas as pd
#     import yaml

#     spath, sfname = None, None

#     if "/" in sfname_or_spath:
#         spath = sfname_or_spath
#     else:
#         sfname = sfname_or_spath

#     if (spath is None) and (sfname is not None):
#         ## for ipython
#         __file__ = inspect.stack()[1].filename
#         if "ipython" in __file__:
#             __file__ = "/tmp/fake.py"

#         ## spath
#         fpath = __file__
#         fdir, fname, _ = split_fpath(fpath)
#         sdir = fdir + fname + "/"
#         spath = sdir + sfname
#         # spath = mk_spath(sfname, makedirs=True)

#     ## Make directory
#     if makedirs:
#         sdir = os.path.dirname(spath)
#         os.makedirs(sdir, exist_ok=True)

#     ## Saves
#     # csv
#     if spath.endswith(".csv"):
#         obj.to_csv(spath)
#     # numpy
#     if spath.endswith(".npy"):
#         np.save(spath, obj)
#     # pkl
#     if spath.endswith(".pkl"):
#         with open(spath, "wb") as s:  # 'w'
#             pickle.dump(obj, s)
#     # png
#     if spath.endswith(".png"):
#         obj.savefig(spath)
#         del obj
#     # tiff
#     if spath.endswith(".tiff") or spath.endswith(".tif"):
#         obj.savefig(spath, dpi=300, format="tiff")  # obj is matplotlib.pyplot object
#         del obj
#     # yaml
#     if spath.endswith(".yaml"):
#         with open(spath, "w") as f:
#             yaml.dump(obj, f)
#     # hdf5
#     if spath.endswith(".hdf5"):
#         name_list, obj_list = []
#         for k, v in obj.items():
#             name_list.append(k)
#             obj_list.append(v)
#         with h5py.File(spath, "w") as hf:
#             for (name, obj) in zip(name_list, obj_list):
#                 hf.create_dataset(name, data=obj)
#     # pth
#     if spath.endswith(".pth"):
#         # torch.save(obj.state_dict(), spath)
#         torch.save(obj, spath)

#     print("\nSaved to: {s}\n".format(s=spath))


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


# def save_listed_scalars_as_csv(
#     listed_scalars,
#     spath_csv,
#     column_name="_",
#     indi_suffix=None,
#     round=3,
#     overwrite=False,
# ):
#     """Puts to df and save it as csv"""
#     if overwrite == True:
#         mv_to_tmp(spath_csv, L=2)
#     indi_suffix = np.arange(len(listed_scalars)) if indi_suffix is None else indi_suffix
#     df = pd.DataFrame(
#         {"{}".format(column_name): listed_scalars}, index=indi_suffix
#     ).round(round)
#     df.to_csv(spath_csv)
#     print("\nSaved to: {}\n".format(spath_csv))


# def save_listed_dfs_as_csv(listed_dfs, spath_csv, indi_suffix=None, overwrite=False):
#     """listed_dfs:
#         [df1, df2, df3, ..., dfN]. They will be written vertically in the order.

#     spath_csv:
#         /hoge/fuga/foo.csv

#     indi_suffix:
#         At the left top cell on the output csv file, '{}'.format(indi_suffix[i])
#         will be added, where i is the index of the df.On the other hand,
#         when indi_suffix=None is passed, only '{}'.format(i) will be added.
#     """
#     if overwrite == True:
#         mv_to_tmp(spath_csv, L=2)

#     indi_suffix = np.arange(len(listed_dfs)) if indi_suffix is None else indi_suffix
#     for i, df in enumerate(listed_dfs):
#         with open(spath_csv, mode="a") as f:
#             f_writer = csv.writer(f)
#             i_suffix = indi_suffix[i]
#             f_writer.writerow(["{}".format(indi_suffix[i])])
#         df.to_csv(spath_csv, mode="a", index=True, header=True)
#         with open(spath_csv, mode="a") as f:
#             f_writer = csv.writer(f)
#             f_writer.writerow([""])
#     print("Saved to: {}".format(spath_csv))


# def mv_to_tmp(fpath, L=2):
#     import os
#     from shutil import move

#     try:
#         tgt_fname = connect_strs_with_hyphens(fpath.split("/")[-L:])
#         tgt_fpath = "/tmp/{}".format(tgt_fname)
#         move(fpath, tgt_fpath)
#         print("Moved to: {}".format(tgt_fpath))
#     except:
#         pass


# # def makedirs_from_spath(spath):
# #     import os
# #     sdir = os.path.dirname(spath)
# #     os.makedirs(sdir, exist_ok=True)
