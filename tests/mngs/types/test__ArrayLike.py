# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import pytest
    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/types/_ArrayLike.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-02-27 11:17:14 (ywatanabe)"
# # File: ./src/mngs/types/_ArrayLike.py
# 
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/types/_ArrayLike.py"
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2025-02-27 11:17:14 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/types/_ArrayLike.py
# 
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/types/_ArrayLike.py"
# 
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-03 02:55:10 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/types/_ArrayLike.py
# 
# from typing import List, Tuple, Union
# 
# import numpy as _np
# import pandas as _pd
# import torch as _torch
# import xarray as _xr
# 
# ArrayLike = Union[
#     List,
#     Tuple,
#     _np.ndarray,
#     _pd.Series,
#     _pd.DataFrame,
#     _xr.DataArray,
#     _torch.tensor,
#     # _torch.Tensor,
# ]
# 
# 
# def is_array_like(obj) -> bool:
#     """Check if object is array-like."""
#     return isinstance(
#         obj,
#         (List, Tuple, _np.ndarray, _pd.Series, _pd.DataFrame, _xr.DataArray),
#     ) or _torch.is_tensor(obj)
# 
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/types/_ArrayLike.py
# --------------------------------------------------------------------------------
