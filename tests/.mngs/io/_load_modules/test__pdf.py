# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-14 07:55:46 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/io/_load_modules/_pdf.py
# 
# def _load_pdf(lpath, **kwargs):
#     """Load PDF file and return extracted text."""
#     try:
#         if not lpath.endswith(".pdf"):
#             raise ValueError("File must have .pdf extension")
# 
#         reader = PyPDF2.PdfReader(lpath)
#         full_text = []
#         for page_num in range(len(reader.pages)):
#             page = reader.pages[page_num]
#             full_text.append(page.extract_text())
#         return "\n".join(full_text)
#     except (ValueError, FileNotFoundError, PyPDF2.PdfReadError) as e:
#         raise ValueError(f"Error loading PDF {lpath}: {str(e)}")
# 
# 
# # EOF

# test from here --------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from mngs..io._load_modules._pdf import *

class Test_MainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        pass

    def teardown_method(self):
        # Clean up after tests
        pass

    def test_basic_functionality(self):
        # Basic test case
        pass

    def test_edge_cases(self):
        # Edge case testing
        pass

    def test_error_handling(self):
        # Error handling testing
        pass
