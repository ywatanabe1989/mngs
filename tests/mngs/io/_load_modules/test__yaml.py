# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import pytest
    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/io/_load_modules/_yaml.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-14 07:41:37 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/io/_load_modules/_yaml.py
# 
# import yaml
# 
# 
# def _load_yaml(lpath, **kwargs):
#     """Load YAML file with optional key lowercasing."""
#     if not lpath.endswith((".yaml", ".yml")):
#         raise ValueError("File must have .yaml or .yml extension")
# 
#     lower = kwargs.pop("lower", False)
#     with open(lpath) as f:
#         obj = yaml.safe_load(f, **kwargs)
# 
#     if lower:
#         obj = {k.lower(): v for k, v in obj.items()}
#     return obj
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/io/_load_modules/_yaml.py
# --------------------------------------------------------------------------------
