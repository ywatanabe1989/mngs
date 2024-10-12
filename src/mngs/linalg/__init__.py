#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-11 08:34:26 (ywatanabe)"

try:
    from ._distance import cdist, euclidean_distance, edist
except ImportError as e:
    print(f"Warning: Failed to import from ._distance.")

try:
    from ._geometric_median import geometric_median
except ImportError as e:
    print(f"Warning: Failed to import geometric_median.")

try:
    from ._misc import cosine, nannorm, rebase_a_vec, three_line_lengths_to_coords
except ImportError as e:
    print(f"Warning: Failed to import from ._misc.")
