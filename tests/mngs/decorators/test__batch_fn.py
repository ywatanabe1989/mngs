# Add your tests here

if __name__ == "__main__":
    import os
    import pytest
    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/decorators/_batch_fn.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-04-24 15:37:59 (ywatanabe)"
# # File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_batch_fn.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/decorators/_batch_fn.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# from functools import wraps
# from typing import Any as _Any
# from typing import Callable
# 
# import torch
# from tqdm import tqdm as _tqdm
# 
# 
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
# 
#     return wrapper
# 
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/decorators/_batch_fn.py
# --------------------------------------------------------------------------------
