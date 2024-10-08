import numpy as _np
import pandas as _pd
import xarray as _xr
import torch as _torch
from typing import List, Tuple, Dict, Any, Union, Sequence, Literal, Optional
from collections.abc import Iterable

ArrayLike = Union[
    List,
    Tuple,
    _np.ndarray,
    _pd.Series,
    _pd.DataFrame,
    _xr.DataArray,
    _torch.Tensor,
]


# from mngs.typing import List, Tuple, Dict, Any, Union, Sequence, Literal, Iterable, ArrayLike
