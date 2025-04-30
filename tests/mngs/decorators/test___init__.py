#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-30 14:24:24 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/decorators/test___init__.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/decorators/test___init__.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import sys

import pytest


def test_init_imports():
    """Test that __init__.py imports all expected decorators."""

    # Force reload of the module to ensure fresh imports
    if "mngs.decorators" in sys.modules:
        del sys.modules["mngs.decorators"]

    # Import the module
    import mngs.decorators

    # Check for expected attributes
    expected_decorators = [
        "cache_disk",
        "cache_mem",
        "preserve_doc",
        "timeout",
        "torch_fn",
        "wrap",
        "numpy_fn",
        "pandas_fn",
        "not_implemented",
    ]

    for decorator in expected_decorators:
        assert hasattr(
            mngs.decorators, decorator
        ), f"Missing decorator: {decorator}"


# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/decorators/__init__.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-04-30 14:51:57 (ywatanabe)"
# # File: /home/ywatanabe/proj/mngs_repo/src/mngs/decorators/__init__.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/decorators/__init__.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# from ._cache_disk import *
# from ._cache_mem import *
# from ._converters import *
# from ._DataTypeDecorators import *
# from ._deprecated import *
# from ._not_implemented import *
# from ._numpy_fn import *
# from ._pandas_fn import *
# from ._preserve_doc import *
# from ._timeout import *
# from ._torch_fn import *
# from ._wrap import *
# from ._batch_fn import *
# 
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/decorators/__init__.py
# --------------------------------------------------------------------------------
