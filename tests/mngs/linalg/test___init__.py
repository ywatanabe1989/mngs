# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/linalg/__init__.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-10-24 18:39:23 (ywatanabe)"
# # mngs_repo/src/mngs/linalg/__init__.py
# 
# from ._distance import cdist, euclidean_distance, edist
# from ._geometric_median import geometric_median
# from ._misc import cosine, nannorm, rebase_a_vec, three_line_lengths_to_coords
# 
# # # Import distance-related functions
# # cdist, euclidean_distance, edist = try_import(
# #     "._distance", ["cdist", "euclidean_distance", "edist"]
# # )
# 
# # # Import geometric median
# # geometric_median, = try_import(
# #     "._geometric_median", ["geometric_median"]
# # )
# 
# # # Import miscellaneous functions
# # cosine, nannorm, rebase_a_vec, three_line_lengths_to_coords = try_import(
# #     "._misc", ["cosine", "nannorm", "rebase_a_vec", "three_line_lengths_to_coords"]
# # )

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/linalg/__init__.py
# --------------------------------------------------------------------------------
