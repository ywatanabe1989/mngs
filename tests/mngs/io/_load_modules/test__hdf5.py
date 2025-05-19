# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_load_modules/_hdf5.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-14 07:55:37 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/io/_load_modules/_hdf5.py
# 
# from typing import Any
# 
# import h5py
# 
# 
# def _load_hdf5(lpath: str, **kwargs) -> Any:
#     """Load HDF5 file."""
#     if not lpath.endswith(".hdf5"):
#         raise ValueError("File must have .hdf5 extension")
#     obj = {}
#     with h5py.File(lpath, "r") as hf:
#         for name in hf:
#             obj[name] = hf[name][:]
#     return obj
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_load_modules/_hdf5.py
# --------------------------------------------------------------------------------
