#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-27 22:09:45 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/tests/mngs/plt/test__add_hue.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/test__add_hue.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import pandas as pd
import pytest
from mngs.plt._add_hue import add_hue

# Source code from: /home/ywatanabe/proj/mngs_dev/src/mngs/plt/_add_hue.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2023-02-11 21:04:35 (ywatanabe)"
#
# import pandas as pd
# import numpy as np
#
# def add_hue(df):
#     df["hue"] = 0
#     dummy_row = pd.DataFrame(
#         columns=df.columns,
#         data=np.array([np.nan for _ in df.columns]).reshape(1, -1),
#     )
#     dummy_row = {}
#     for col in df.columns:
#         dtype = df[col].dtype
#         if dtype is np.dtype(object):
#             dummy_row[col] = np.nan
#         if dtype is np.dtype(float):
#             dummy_row[col] = np.nan
#         if dtype is np.dtype(np.int64):
#             dummy_row[col] = np.nan
#         if dtype is np.dtype(bool):
#             dummy_row[col] = None
#
#     dummy_row = pd.DataFrame(pd.Series(dummy_row)).T
#
#     dummy_row["hue"] = 1
#     df_added = pd.concat([df, dummy_row], axis=0)
#     return df_added


def test_add_hue_appends_dummy_row():
    df = pd.DataFrame({"col1": [1, 2]})
    df2 = add_hue(df)
    assert "hue" in df2.columns
    assert df2["hue"].iloc[-1] == 1
    assert len(df2) == len(df) + 1


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])

# EOF