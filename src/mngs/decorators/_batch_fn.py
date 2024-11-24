#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-17 13:05:41 (ywatanabe)"
# File: ./mngs_repo/src/mngs/decorators/_batch_fn.py

__file__ = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_batch_fn.py"

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-17 12:49:32 (ywatanabe)"
# File: ./mngs_repo/src/mngs/decorators/_batch_fn.py

from typing import List, Tuple, Union

__file__ = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_batch_fn.py"

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-17 12:07:55 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/decorators/_batch_fn.py

# __file__ = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_batch_fn.py"

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-04 02:56:44 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/decorators/_batch_fn.py

# from functools import wraps
# from typing import Any as _Any
# from typing import Callable, List, Tuple, Union

# import torch
# from tqdm import tqdm as _tqdm

# from ._converters import (_conversion_warning, _try_device, is_torch, to_numpy,
#                          to_torch)

# def batch_fn(func: Callable) -> Callable:
#     """Processes input data in batches to handle memory constraints.

#     Example
#     -------
#     >>> @batch_fn
#     ... def process_data(data, param1=None):
#     ...     return model(data)
#     >>> result = process_data(large_data, batch_size=32)

#     Parameters
#     ----------
#     func : Callable
#         Function to be decorated for batch processing

#     Returns
#     -------
#     Callable
#         Wrapped function that handles batch processing

#     Notes
#     -----
#     Automatically handles torch.Tensor and tuple returns
#     """
#     @wraps(func)
#     def wrapper(
#         x: Union[List, torch.Tensor],
#         *args: _Any,
#         **kwargs: _Any
#     ) -> Union[List, torch.Tensor, Tuple[torch.Tensor, ...]]:
#         batch_size = int(kwargs.pop("batch_size", 4))

#         if not hasattr(x, '__len__'):
#             raise TypeError("Input must be a sequence with length")

#         if len(x) <= batch_size:
#             return func(x, *args, **kwargs, batch_size=batch_size)

#         n_batches = (len(x) + batch_size - 1) // batch_size
#         results = []

#         for i_batch in _tqdm(range(n_batches), desc="Processing batches"):
#             start = i_batch * batch_size
#             end = min((i_batch + 1) * batch_size, len(x))
#             batch_result = func(x[start:end], *args, **kwargs, batch_size=batch_size)

#             # Handle different return types
#             if isinstance(batch_result, torch.Tensor):
#                 batch_result = batch_result.cpu()
#             elif isinstance(batch_result, tuple):
#                 batch_result = tuple(
#                     val.cpu() if isinstance(val, torch.Tensor) else val
#                     for val in batch_result
#                 )
#             results.append(batch_result)

#         # Combine results based on return type
#         if isinstance(results[0], tuple):
#             n_vars = len(results[0])
#             try:
#                 combined_results = [
#                     torch.vstack([res[i_var] for res in results])
#                     for i_var in range(n_vars)
#                 ]
#                 return tuple(combined_results)
#             except RuntimeError as e:
#                 raise RuntimeError(f"Failed to combine tuple results: {e}")
#         elif isinstance(results[0], torch.Tensor):
#             try:
#                 return torch.vstack(results)
#             except RuntimeError as e:
#                 raise RuntimeError(f"Failed to combine tensor results: {e}")
#         else:
#             return sum(results, [])

#     return wrapper

# # EOF

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-17 12:07:20 (ywatanabe)"
# File: ./mngs_repo/src/mngs/decorators/_batch_fn.py

__file__ = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_batch_fn.py"

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-04 02:56:44 (ywatanabe)"
# File: ./mngs_repo/src/mngs/decorators/_batch_fn.py

from functools import wraps
from typing import Any as _Any
from typing import Callable

import torch
from tqdm import tqdm as _tqdm

# def batch_fn(func: Callable) -> Callable:
#     @wraps(func)
#     def wrapper(x: _Any, *args: _Any, **kwargs: _Any) -> _Any:
#         batch_size = int(kwargs.pop("batch_size", 4))
#         if len(x) <= batch_size:
#             return func(x, *args, **kwargs, batch_size=batch_size)
#         n_batches = (len(x) + batch_size - 1) // batch_size
#         results = []
#         for i_batch in _tqdm(range(n_batches)):
#             start = i_batch * batch_size
#             end = min((i_batch + 1) * batch_size, len(x))
#             batch_result = func(
#                 x[start:end], *args, **kwargs, batch_size=batch_size
#             )
#             if isinstance(batch_result, torch.Tensor):
#                 batch_result = batch_result.cpu()
#             elif isinstance(batch_result, tuple):
#                 batch_result = tuple(
#                     val.cpu() if isinstance(val, torch.Tensor) else val
#                     for val in batch_result
#                 )
#             results.append(batch_result)
#         if isinstance(results[0], tuple):
#             n_vars = len(results[0])
#             combined_results = [
#                 torch.vstack([res[i_var] for res in results])
#                 for i_var in range(n_vars)
#             ]
#             return tuple(combined_results)
#         elif isinstance(results[0], torch.Tensor):
#             return torch.vstack(results)
#         else:
#             return sum(results, [])

#     return wrapper


# def batch_fn(func: Callable) -> Callable:
#     @wraps(func)
#     def wrapper(
#         x: Union[List, torch.Tensor], *args: _Any, **kwargs: _Any
#     ) -> Union[List, torch.Tensor, Tuple[torch.Tensor, ...]]:
#         batch_size = int(kwargs.pop("batch_size", 4))
#         if len(x) <= batch_size:
#             return func(x, *args, **kwargs, batch_size=batch_size)
#         n_batches = (len(x) + batch_size - 1) // batch_size
#         results = []
#         for i_batch in _tqdm(
#             range(n_batches),
#             desc=f"Processing {len(x)} items in batches of {batch_size}",
#         ):
#             start = i_batch * batch_size
#             end = min((i_batch + 1) * batch_size, len(x))
#             batch_result = func(
#                 x[start:end], *args, **kwargs, batch_size=batch_size
#             )
#             if isinstance(batch_result, torch.Tensor):
#                 batch_result = batch_result.cpu()
#             elif isinstance(batch_result, tuple):
#                 batch_result = tuple(
#                     val.cpu() if isinstance(val, torch.Tensor) else val
#                     for val in batch_result
#                 )
#             results.append(batch_result)
#         if isinstance(results[0], tuple):
#             n_vars = len(results[0])
#             combined_results = [
#                 torch.vstack([res[i_var] for res in results])
#                 for i_var in range(n_vars)
#             ]
#             return tuple(combined_results)
#         elif isinstance(results[0], torch.Tensor):
#             return torch.vstack(results)
#         else:
#             return sum(results, [])

#     return wrapper

def batch_fn(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(
        x: Union[List, torch.Tensor], *args: _Any, **kwargs: _Any
    ) -> Union[List, torch.Tensor, Tuple[torch.Tensor, ...]]:
        batch_size = int(kwargs.pop("batch_size", 4))

        print(f"Batch size: {batch_size} (determined by @batch_fn decorator)")
        if len(x) <= batch_size:
            return func(x, *args, **kwargs, batch_size=batch_size)

        n_batches = (len(x) + batch_size - 1) // batch_size
        results = []

        for i_batch in _tqdm(
            range(n_batches),
            desc=f"Processing {len(x)} items in batches of {batch_size}",
        ):
            start = i_batch * batch_size
            end = min((i_batch + 1) * batch_size, len(x))
            batch_result = func(
                x[start:end], *args, **kwargs, batch_size=batch_size
            )
            results.append(batch_result)

        # Separate tensors and names
        tensors = [r[0] for r in results]
        names = results[0][1]

        return torch.vstack(tensors), names

    return wrapper


# EOF
