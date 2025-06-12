#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-12-06 10:30:00"
# Author: Claude
# Filename: __init__.py

"""
MNGS Scholar: Scientific literature search and management.

A unified interface for searching scientific papers from both web sources
(PubMed, arXiv, Semantic Scholar) and local PDF collections, with support
for vector-based semantic search.

Examples
--------
>>> import mngs.scholar
>>> 
>>> # Simple search (web only by default)
>>> papers = mngs.scholar.search_sync("deep learning sleep")
>>> 
>>> # Search both web and specific local directories
>>> papers = mngs.scholar.search_sync(
...     "transformer architecture",
...     local=["./papers", "~/Documents/research"]
... )
>>> 
>>> # Local-only search
>>> papers = mngs.scholar.search_sync(
...     "neural oscillations",
...     web=False,
...     local=["./my_papers"]
... )
>>> 
>>> # Web search with PDF download
>>> papers = mngs.scholar.search_sync(
...     "attention mechanism",
...     download_pdfs=True
... )
>>> 
>>> # Build index for faster local search
>>> stats = mngs.scholar.build_index(["./papers"])
"""

from ._search import search, search_sync, build_index, get_scholar_dir
from ._paper import Paper
from ._vector_search import VectorSearchEngine
from ._web_sources import search_pubmed, search_arxiv, search_semantic_scholar, search_all_sources
from ._local_search import LocalSearchEngine
from ._pdf_downloader import PDFDownloader


__all__ = [
    # Main API
    "search",
    "search_sync",
    "build_index",
    "get_scholar_dir",
    
    # Core classes
    "Paper",
    "VectorSearchEngine",
    "LocalSearchEngine",
    "PDFDownloader",
    
    # Web search functions
    "search_pubmed",
    "search_arxiv",
    "search_semantic_scholar",
    "search_all_sources",
]


# Configure default logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())