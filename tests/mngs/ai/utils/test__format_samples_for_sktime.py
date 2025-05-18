# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/ai/utils/_format_samples_for_sktime.py
# --------------------------------------------------------------------------------
# import pandas as pd
# import torch
# import numpy as np
# 
# def _format_a_sample_for_sktime(x):
#     """
#     x.shape: (n_chs, seq_len)
#     """
#     dims = pd.Series(
#         [pd.Series(x[d], name=f"dim_{d}") for d in range(len(x))],
#         index=[f"dim_{i}" for i in np.arange(len(x))]
#     )
#     return dims
# 
# def format_samples_for_sktime(X):
#     """
#     X.shape: (n_samples, n_chs, seq_len)
#     """
#     if torch.is_tensor(X):
#         X = X.numpy() # (64, 160, 1024)
# 
#         X = X.astype(np.float64)
# 
#     return pd.DataFrame(
#         [_format_a_sample_for_sktime(X[i]) for i in range(len(X))]
#         )

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/ai/utils/_format_samples_for_sktime.py
# --------------------------------------------------------------------------------
