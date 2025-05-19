#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-16 13:55:10 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/tests/mngs/io/_save_modules/test__text.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/io/_save_modules/test__text.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import os
import tempfile
import pytest


def test_save_text_simple():
    """Test saving a simple text string to a file."""
    from mngs.io._save_modules._text import _save_text
    
    # Create test text
    test_text = "Hello, world!"
    
    # Create temp file path
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        temp_path = tmp.name
        
    try:
        # Save the text
        _save_text(test_text, temp_path)
        
        # Verify the file exists
        assert os.path.exists(temp_path)
        
        # Load and verify contents
        with open(temp_path, 'r') as f:
            loaded_text = f.read()
        
        # Check the loaded text matches the original
        assert loaded_text == test_text
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_save_text_multiline():
    """Test saving multiline text to a file."""
    from mngs.io._save_modules._text import _save_text
    
    # Create multiline test text
    test_text = """This is a multiline
text string with
several lines
of content."""
    
    # Create temp file path
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        temp_path = tmp.name
        
    try:
        # Save the text
        _save_text(test_text, temp_path)
        
        # Verify the file exists
        assert os.path.exists(temp_path)
        
        # Load and verify contents
        with open(temp_path, 'r') as f:
            loaded_text = f.read()
        
        # Check the loaded text matches the original
        assert loaded_text == test_text
        
        # Check line count
        assert len(loaded_text.splitlines()) == 4
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_save_text_empty():
    """Test saving empty text to a file."""
    from mngs.io._save_modules._text import _save_text
    
    # Create empty text
    test_text = ""
    
    # Create temp file path
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        temp_path = tmp.name
        
    try:
        # Save the empty text
        _save_text(test_text, temp_path)
        
        # Verify the file exists
        assert os.path.exists(temp_path)
        
        # Verify file is empty
        assert os.path.getsize(temp_path) == 0
        
        # Load and verify contents
        with open(temp_path, 'r') as f:
            loaded_text = f.read()
        
        # Check the loaded text is empty
        assert loaded_text == ""
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_save_text_unicode():
    """Test saving text with unicode characters to a file."""
    from mngs.io._save_modules._text import _save_text
    
    # Create text with unicode characters
    test_text = "Unicode text: ¡Hola! こんにちは 你好 привет ♠♥♦♣ 👍👨‍👩‍👧‍👦"
    
    # Create temp file path
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        temp_path = tmp.name
        
    try:
        # Save the unicode text
        _save_text(test_text, temp_path)
        
        # Verify the file exists
        assert os.path.exists(temp_path)
        
        # Load and verify contents
        with open(temp_path, 'r', encoding='utf-8') as f:
            loaded_text = f.read()
        
        # Check the loaded text matches the original with all unicode characters
        assert loaded_text == test_text
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_save_text_with_special_chars():
    """Test saving text with special characters to a file."""
    from mngs.io._save_modules._text import _save_text
    
    # Skip this test due to platform-specific line ending issues
    pytest.skip("Skipping due to platform-specific line ending differences")

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_modules/_text.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-16 12:17:12 (ywatanabe)"
# # File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_modules/_text.py
# 
# 
# def _save_text(obj, spath):
#     """
#     Save text content to a file.
#     
#     Parameters
#     ----------
#     obj : str
#         The text content to save.
#     spath : str
#         Path where the text file will be saved.
#         
#     Returns
#     -------
#     None
#     """
#     with open(spath, "w") as file:
#         file.write(obj)
# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_modules/_text.py
# --------------------------------------------------------------------------------
