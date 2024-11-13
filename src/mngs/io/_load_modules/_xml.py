#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-14 07:55:49 (ywatanabe)"
# File: ./mngs_repo/src/mngs/io/_load_modules/_xml.py

def _load_xml(lpath, **kwargs):
    """Load XML file and convert to dict."""
    if not lpath.endswith(".xml"):
        raise ValueError("File must have .xml extension")
    return xml2dict(lpath, **kwargs)


# EOF
