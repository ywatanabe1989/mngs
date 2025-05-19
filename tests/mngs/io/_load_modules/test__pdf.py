# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_load_modules/_pdf.py
# --------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_load_modules/_pdf.py
# --------------------------------------------------------------------------------
