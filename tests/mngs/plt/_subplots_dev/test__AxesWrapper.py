# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_subplots_dev/_AxesWrapper.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-04-29 19:56:30 (ywatanabe)"
# # File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/_subplots/_AxesWrapper.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/plt/_subplots/_AxesWrapper.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# import pandas as pd
# 
# 
# class AxesWrapper:
#     def __init__(self, fig_mngs, axes_mngs):
#         self._fig_mngs = fig_mngs
#         self._axes_mngs = axes_mngs
# 
#     def get_figure(self):
#         return self._fig_mngs
# 
#     # def __getattr__(self, name):
#     #     print(f"Attribute of AxesWrapper: {name}")
#     #     methods = []
#     #     try:
#     #         for axis in self._axes_mngs.flat:
#     #             methods.append(getattr(axis, name))
#     #     except Exception:
#     #         methods = []
# 
#     #     if methods and all(callable(m) for m in methods):
# 
#     #         @wraps(methods[0])
#     #         def wrapper(*args, **kwargs):
#     #             return [
#     #                 getattr(ax, name)(*args, **kwargs)
#     #                 for ax in self._axes_mngs.flat
#     #             ]
# 
#     #         return wrapper
# 
#     #     if methods and not callable(methods[0]):
#     #         return methods
# 
#     #     warnings.warn(
#     #         f"MNGS AxesWrapper: '{name}' not implemented, ignored.",
#     #         UserWarning,
#     #     )
# 
#     #     def dummy(*args, **kwargs):
#     #         return None
# 
#     #     return dummy
# 
#     def __getitem__(self, index):
#         subset = self._axes_mngs[index]
#         if isinstance(index, slice):
#             return AxesWrapper(self._fig_mngs, subset)
#         return subset
# 
#     def __iter__(self):
#         return iter(self._axes_mngs.flat)
# 
#     def __len__(self):
#         return self._axes_mngs.size
# 
#     def legend(self, loc="upper left"):
#         return [ax.legend(loc=loc) for ax in self._axes_mngs.flat]
# 
#     @property
#     def history(self):
#         return [ax.history for ax in self._axes_mngs.flat]
# 
#     @property
#     def shape(self):
#         return self._axes_mngs.shape
# 
#     def export_as_csv(self):
#         dfs = []
#         for ii, ax in enumerate(self._axes_mngs.flat):
#             df = ax.export_as_csv()
#             df.columns = [f"ax_{ii:02d}_{col}" for col in df.columns]
#             dfs.append(df)
#         return pd.concat(dfs, axis=1) if dfs else pd.DataFrame()
# 
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_subplots_dev/_AxesWrapper.py
# --------------------------------------------------------------------------------
