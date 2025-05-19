#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-02 09:02:24 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/ax/_adjust/test__format_label.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/ax/_adjust/test__format_label.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

from mngs.plt.ax._style._format_label import format_label


class TestMainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        pass

    def teardown_method(self):
        # Clean up after tests
        pass

    def test_basic_functionality(self):
        # Current implementation just returns the input label
        # String inputs
        assert format_label("test_label") == "test_label"
        assert (
            format_label("complex_label_with_underscores")
            == "complex_label_with_underscores"
        )
        assert format_label("UPPERCASE") == "UPPERCASE"

        # Non-string inputs
        assert format_label(123) == 123
        assert format_label(None) is None
        assert format_label([1, 2, 3]) == [1, 2, 3]

    def test_commented_functionality(self):
        # Test the currently commented out functionality
        # These tests would be used if the commented code is uncommented

        # The following assertions would pass if the commented functionality was active:
        # assert format_label("test_label") == "Test Label"
        # assert format_label("complex_label_with_underscores") == "Complex Label With Underscores"
        # assert format_label("UPPERCASE") == "UPPERCASE"  # Should remain uppercase

        # For now, we just test the actual behavior
        assert format_label("test_label") == "test_label"
        assert (
            format_label("complex_label_with_underscores")
            == "complex_label_with_underscores"
        )
        assert format_label("UPPERCASE") == "UPPERCASE"

    def test_edge_cases(self):
        # Empty string
        assert format_label("") == ""

        # String with special characters
        assert format_label("special!@#$%^&*()_+") == "special!@#$%^&*()_+"

        # Unicode characters
        assert format_label("unicode_текст_测试") == "unicode_текст_测试"

    def test_savefig(self):
        import matplotlib.pyplot as plt
        from mngs.io import save

        # Setup
        fig, ax = plt.figure(), plt.subplot(111)
        ax.plot([1, 2, 3], [1, 2, 3])

        # Main test functionality
        label = format_label("test_label")
        ax.set_title(label)

        # Saving
        spath = f"./{os.path.basename(__file__)}.jpg"
        save(fig, spath)

        # Check saved file
        ACTUAL_SAVE_DIR = __file__.replace(".py", "_out")
        actual_spath = os.path.join(ACTUAL_SAVE_DIR, spath)
        assert os.path.exists(
            actual_spath
        ), f"Failed to save figure to {spath}"

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/plt/ax/_style/_format_label.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-09-15 09:39:02 (ywatanabe)"
# # /home/ywatanabe/proj/_mngs_repo_openhands/src/mngs/plt/ax/_format_label.py
# 
# 
# def format_label(label):
#     """
#     Format label by capitalizing first letter and replacing underscores with spaces.
#     """
# 
#     # if isinstance(label, str):
#     #     # Replace underscores with spaces
#     #     label = label.replace("_", " ")
# 
#     #     # Capitalize first letter of each word
#     #     label = " ".join(word.capitalize() for word in label.split())
# 
#     #     # Special case for abbreviations (all caps)
#     #     if label.isupper():
#     #         return label
# 
#     return label

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/plt/ax/_style/_format_label.py
# --------------------------------------------------------------------------------
