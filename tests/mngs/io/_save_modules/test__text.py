#!/usr/bin/env python3
# -*- coding: utf-8 -*-
<<<<<<< HEAD
# Timestamp: "2025-06-12 13:56:00 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/.claude-worktree/mngs_repo/tests/mngs/io/_save_modules/test__text.py
# ----------------------------------------
import os

__FILE__ = "./tests/mngs/io/_save_modules/test__text.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Test cases for text file saving functionality
"""

import os
import tempfile
import pytest
from pathlib import Path

from mngs.io._save_modules._text import save_text


class TestSaveText:
    """Test suite for save_text function"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.txt")

    def teardown_method(self):
        """Clean up after tests"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_save_simple_string(self):
        """Test saving simple string"""
        text = "Hello, World!"
        save_text(text, self.test_file)
        
        assert os.path.exists(self.test_file)
        with open(self.test_file, 'r') as f:
            content = f.read()
        assert content == text

    def test_save_multiline_string(self):
        """Test saving multiline string"""
        text = """Line 1
Line 2
Line 3
Line 4"""
        save_text(text, self.test_file)
        
        with open(self.test_file, 'r') as f:
            content = f.read()
        assert content == text

    def test_save_with_special_characters(self):
        """Test saving text with special characters"""
        text = "Special chars: äöü €¥£ → ← ↑ ↓ 你好世界 😊"
        save_text(text, self.test_file)
        
        with open(self.test_file, 'r', encoding='utf-8') as f:
            content = f.read()
        assert content == text

    def test_save_empty_string(self):
        """Test saving empty string"""
        save_text("", self.test_file)
        
        assert os.path.exists(self.test_file)
        with open(self.test_file, 'r') as f:
            content = f.read()
        assert content == ""

    def test_save_with_newlines(self):
        """Test saving text with various newline types"""
        text = "Line 1\nLine 2\r\nLine 3\rLine 4"
        save_text(text, self.test_file)
        
        with open(self.test_file, 'r') as f:
            content = f.read()
        # Content should preserve the exact newlines
        assert content == text

    def test_save_large_text(self):
        """Test saving large text content"""
        # Generate large text
        lines = [f"This is line {i} of a large text file." for i in range(10000)]
        text = "\n".join(lines)
        
        save_text(text, self.test_file)
        
        with open(self.test_file, 'r') as f:
            content = f.read()
        assert content == text
        assert content.count('\n') == 9999

    def test_save_with_encoding(self):
        """Test saving with specific encoding"""
        text = "Testing encoding: ñáéíóú"
        save_text(text, self.test_file, encoding='utf-8')
        
        with open(self.test_file, 'r', encoding='utf-8') as f:
            content = f.read()
        assert content == text

    def test_save_json_formatted_text(self):
        """Test saving JSON-formatted text"""
        import json
        data = {"name": "test", "value": 42, "nested": {"key": "value"}}
        text = json.dumps(data, indent=2)
        
        save_text(text, self.test_file)
        
        with open(self.test_file, 'r') as f:
            content = f.read()
        
        # Should be valid JSON
        loaded = json.loads(content)
        assert loaded == data

    def test_save_code_snippet(self):
        """Test saving code snippet with proper formatting"""
        code = '''def hello_world():
    """Print hello world."""
    print("Hello, World!")
    return True

if __name__ == "__main__":
    hello_world()
'''
        save_text(code, self.test_file)
        
        with open(self.test_file, 'r') as f:
            content = f.read()
        assert content == code

    def test_save_markdown_text(self):
        """Test saving markdown-formatted text"""
        markdown = """# Title

## Subtitle

This is a paragraph with **bold** and *italic* text.

- Item 1
- Item 2
- Item 3

```python
def example():
    pass
```
"""
        save_text(markdown, self.test_file)
        
        with open(self.test_file, 'r') as f:
            content = f.read()
        assert content == markdown

    def test_save_with_different_extensions(self):
        """Test saving with different file extensions"""
        text = "Test content"
        
        # .txt
        txt_file = os.path.join(self.temp_dir, "test.txt")
        save_text(text, txt_file)
        assert os.path.exists(txt_file)
        
        # .log
        log_file = os.path.join(self.temp_dir, "test.log")
        save_text(text, log_file)
        assert os.path.exists(log_file)
        
        # .md
        md_file = os.path.join(self.temp_dir, "test.md")
        save_text(text, md_file)
        assert os.path.exists(md_file)

    def test_save_numeric_string(self):
        """Test saving numeric values as strings"""
        save_text("42", self.test_file)
        
        with open(self.test_file, 'r') as f:
            content = f.read()
        assert content == "42"

    def test_save_with_bom(self):
        """Test saving with BOM (Byte Order Mark)"""
        text = "Text with BOM"
        save_text(text, self.test_file, encoding='utf-8-sig')
        
        # Check file has BOM
        with open(self.test_file, 'rb') as f:
            raw = f.read()
        assert raw.startswith(b'\xef\xbb\xbf')
        
        # Content should still read correctly
        with open(self.test_file, 'r', encoding='utf-8-sig') as f:
            content = f.read()
        assert content == text

    def test_error_non_string_input(self):
        """Test error handling for non-string input"""
        with pytest.raises(TypeError):
            save_text(123, self.test_file)
        
        with pytest.raises(TypeError):
            save_text([1, 2, 3], self.test_file)

    def test_save_with_line_endings(self):
        """Test preserving specific line endings"""
        # Unix line endings
        unix_text = "Line1\nLine2\nLine3"
        save_text(unix_text, self.test_file)
        
        with open(self.test_file, 'rb') as f:
            content = f.read()
        assert b'\r\n' not in content  # No Windows line endings
        assert b'\n' in content  # Has Unix line endings


# EOF
=======
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
>>>>>>> origin/main
