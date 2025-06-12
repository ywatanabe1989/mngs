# Release Notes - MNGS v1.12.0

**Release Date**: 2024-12-06

## 🎉 Major Features

### New Scholar Module for Scientific Literature Search
We're excited to introduce `mngs.scholar`, a powerful unified interface for searching and managing scientific literature!

#### Key Features:
- **Unified Search**: Search across PubMed, arXiv, and Semantic Scholar with a single command
- **Local Integration**: Seamlessly search your personal PDF collection alongside web sources
- **Semantic Search**: Find relevant papers using vector-based similarity search
- **Auto Downloads**: Automatically download and organize PDFs from available sources
- **Smart Organization**: Configure storage location with `MNGS_SCHOLAR_DIR` environment variable
- **Citation Export**: Built-in BibTeX generation for easy citations

#### Usage:
```python
import mngs.scholar

# Simple web search
papers = mngs.scholar.search_sync("deep learning")

# Search with local PDF directories
papers = mngs.scholar.search_sync(
    "neural networks",
    local=["./papers", "~/Documents/research"]
)

# Build index for faster local searches
stats = mngs.scholar.build_index(["./papers"])
```

## 🐛 Bug Fixes

### Fixed pip install failures
- Removed symbolic links that were causing setuptools build errors
- Affected files: `dsp/nn`, `decorators/_DataTypeDecorators.py`, `stats/tests/_corr_test.py`, `str/_gen_timestamp.py`, `str/_gen_ID.py`
- pip install now works correctly for all installation methods

## 📝 API Changes

### Scholar Module API
- **New**: `mngs.scholar.search()` - Async search function
- **New**: `mngs.scholar.search_sync()` - Synchronous search wrapper
- **New**: `mngs.scholar.build_index()` - Build local search index
- **New**: `mngs.scholar.Paper` - Paper class with metadata and methods
- **New**: `mngs.scholar.get_scholar_dir()` - Get/set scholar directory

### Breaking Changes
None in this release. The scholar module is a new addition.

## 📚 Documentation

- Comprehensive README for scholar module at `src/mngs/scholar/README.md`
- New example: `examples/mngs/scholar/basic_search_example.py`
- API documentation updated

## 🔧 Dependencies

Optional dependencies for full scholar functionality:
- `sentence-transformers` - For vector search
- `aiohttp` - For async web requests
- `pymupdf` or `PyPDF2` - For PDF text extraction

## 🙏 Acknowledgments

Special thanks to all contributors and users who provided feedback for the scholar module design!

## 📈 What's Next

- Additional web sources (Google Scholar, bioRxiv)
- Full-text search within PDFs
- Citation graph analysis
- Integration with reference managers

---

For questions or issues, please visit: https://github.com/ywatanabe1989/mngs/issues