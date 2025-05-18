#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-01 16:47:30 (ywatanabe)"
# File: /home/ywatanabe/proj/_mngs_repo/tests/mngs/plt/_subplots/_AxisWrapperMixins/test___init__.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/_subplots/_AxisWrapperMixins/test___init__.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
def test_imports():
    """Test that all required mixins are properly exported from the module."""
    # Import the package
    from mngs.plt._subplots._AxisWrapperMixins import (
        AdjustmentMixin,
        MatplotlibPlotMixin,
        SeabornMixin,
        TrackingMixin
    )

    # Verify each mixin is properly imported and is a class
    assert isinstance(AdjustmentMixin, type)
    assert isinstance(MatplotlibPlotMixin, type)
    assert isinstance(SeabornMixin, type)
    assert isinstance(TrackingMixin, type)

def test_import_consistency():
    """Test that module exports match the imports in __init__.py"""
    # Import the package
    import mngs.plt._subplots._AxisWrapperMixins as mixins

    # Get all exported names that don't start with underscore
    exported_names = [name for name in dir(mixins) if not name.startswith('_')]

    # Expected exported names based on the source code
    expected_exports = [
        'AdjustmentMixin',
        'MatplotlibPlotMixin',
        'SeabornMixin',
        'TrackingMixin'
    ]

    # Verify all expected names are exported
    for name in expected_exports:
        assert name in exported_names, f"Expected export '{name}' not found"

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/plt/_subplots/_AxisWrapperMixins/__init__.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-01 08:54:38 (ywatanabe)"
# # File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/_subplots/_AxisWrapperMixins/__init__.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/plt/_subplots/_AxisWrapperMixins/__init__.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# from ._AdjustmentMixin import AdjustmentMixin
# from ._MatplotlibPlotMixin import MatplotlibPlotMixin
# from ._SeabornMixin import SeabornMixin
# from ._TrackingMixin import TrackingMixin
# 
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/plt/_subplots/_AxisWrapperMixins/__init__.py
# --------------------------------------------------------------------------------
