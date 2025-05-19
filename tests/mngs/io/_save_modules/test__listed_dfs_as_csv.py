# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_listed_dfs_as_csv.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 21:28:56 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/io/_save_listed_dfs_as_csv.py
# 
# import csv
# import numpy as np
# from ._mv_to_tmp import _mv_to_tmp
# 
# def _save_listed_dfs_as_csv(
#     listed_dfs,
#     spath_csv,
#     indi_suffix=None,
#     overwrite=False,
#     verbose=False,
# ):
#     """listed_dfs:
#         [df1, df2, df3, ..., dfN]. They will be written vertically in the order.
# 
#     spath_csv:
#         /hoge/fuga/foo.csv
# 
#     indi_suffix:
#         At the left top cell on the output csv file, '{}'.format(indi_suffix[i])
#         will be added, where i is the index of the df.On the other hand,
#         when indi_suffix=None is passed, only '{}'.format(i) will be added.
#     """
# 
#     if overwrite == True:
#         _mv_to_tmp(spath_csv, L=2)
# 
#     indi_suffix = (
#         np.arange(len(listed_dfs)) if indi_suffix is None else indi_suffix
#     )
#     for i, df in enumerate(listed_dfs):
#         with open(spath_csv, mode="a") as f:
#             f_writer = csv.writer(f)
#             i_suffix = indi_suffix[i]
#             f_writer.writerow(["{}".format(indi_suffix[i])])
#         df.to_csv(spath_csv, mode="a", index=True, header=True)
#         with open(spath_csv, mode="a") as f:
#             f_writer = csv.writer(f)
#             f_writer.writerow([""])
#     if verbose:
#         print("Saved to: {}".format(spath_csv))
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_listed_dfs_as_csv.py
# --------------------------------------------------------------------------------
