#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-02 18:03:43 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/color/test__add_hue_col.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/color/test__add_hue_col.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

def test_add_hue_appends_dummy_row():
    import pandas as pd
    from mngs.plt.color._add_hue_col import add_hue_col

    df = pd.DataFrame({"col1": [1, 2]})
    df2 = add_hue_col(df)
    assert "hue" in df2.columns
    assert df2["hue"].iloc[-1] == 1
    assert len(df2) == len(df) + 1

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/plt/color/_add_hue_col.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-02 18:02:24 (ywatanabe)"
# # File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/color/_add_hue_col.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/plt/color/_add_hue_col.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# import numpy as np
# import pandas as pd
# 
# 
# def add_hue_col(df):
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
# 
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/plt/color/_add_hue_col.py
# --------------------------------------------------------------------------------
