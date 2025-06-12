# MNGS Scholar Module

A unified interface for searching and managing scientific literature from both web sources and local collections.

## Features

- **Unified Search Interface**: Search across multiple sources with a single API
- **Web Sources**: PubMed, arXiv, Semantic Scholar
- **Local Search**: Search through your PDF collection
- **Vector Search**: Semantic similarity search using embeddings
- **PDF Management**: Automatic PDF download and organization
- **Caching**: Intelligent caching for faster repeated searches
- **Export**: BibTeX generation for citations

## Quick Start

```python
import mngs.scholar

# Simple search (both web and local)
papers = mngs.scholar.search_sync("deep learning sleep")

# Web-only search with PDF download
papers = mngs.scholar.search_sync(
    "transformer architecture",
    local=False,
    download_pdfs=True
)

# Local-only search
papers = mngs.scholar.search_sync(
    "neural oscillations",
    web=False,
    local_paths=["./papers", "~/Documents/research"]
)
```

## Installation

The scholar module requires some optional dependencies:

```bash
# For PDF text extraction
pip install pymupdf  # or PyPDF2

# For vector search
pip install sentence-transformers

# For web scraping
pip install aiohttp
```

## Configuration

Set the default scholar directory using environment variable:

```bash
export MNGS_SCHOLAR_DIR="~/my_papers"  # Default: ~/.mngs/scholar
```

## API Reference

### Main Functions

#### `search(query, web=True, local=True, ...)`
Asynchronous search function for finding papers.

**Parameters:**
- `query` (str): Search query
- `web` (bool): Search web sources
- `local` (bool): Search local files
- `local_paths` (list): Directories to search
- `max_results` (int): Maximum results to return
- `download_pdfs` (bool): Download PDFs for web results
- `use_vector_search` (bool): Use semantic similarity
- `web_sources` (list): Web sources to search

#### `search_sync(...)`
Synchronous wrapper for the search function.

#### `build_index(paths, recursive=True, build_vector_index=True)`
Build search index for local papers.

**Parameters:**
- `paths` (list): Directories to index
- `recursive` (bool): Search subdirectories
- `build_vector_index` (bool): Create vector embeddings

### Classes

#### `Paper`
Represents a scientific paper with metadata.

**Attributes:**
- `title`: Paper title
- `authors`: List of authors
- `abstract`: Paper abstract
- `year`: Publication year
- `doi`: Digital Object Identifier
- `source`: Source (pubmed, arxiv, local, etc.)
- `pdf_path`: Path to local PDF

**Methods:**
- `to_bibtex()`: Generate BibTeX entry
- `has_pdf()`: Check if PDF is available
- `get_identifier()`: Get unique identifier

## Examples

### Building a Local Index

```python
import mngs.scholar

# Index your paper collection
stats = mngs.scholar.build_index([
    "./papers",
    "~/Documents/research"
])

print(f"Indexed {stats['local_files_indexed']} files")
```

### Advanced Search with Filters

```python
import asyncio
import mngs.scholar

async def advanced_search():
    # Search specific sources
    papers = await mngs.scholar.search(
        "neural networks",
        web_sources=["arxiv", "pubmed"],
        max_results=20,
        use_vector_search=True
    )
    
    # Filter by year
    recent_papers = [p for p in papers if p.year and p.year >= 2020]
    
    return recent_papers

papers = asyncio.run(advanced_search())
```

### Exporting Citations

```python
# Get papers
papers = mngs.scholar.search_sync("transformer attention")

# Export as BibTeX
with open("references.bib", "w") as f:
    for paper in papers:
        f.write(paper.to_bibtex())
        f.write("\n\n")
```

## Architecture

The scholar module consists of several components:

1. **Search Interface** (`_search.py`): Main entry point
2. **Paper Class** (`_paper.py`): Paper representation
3. **Web Sources** (`_web_sources.py`): API integrations
4. **Local Search** (`_local_search.py`): PDF indexing and search
5. **Vector Search** (`_vector_search.py`): Semantic similarity
6. **PDF Downloader** (`_pdf_downloader.py`): Automatic downloads

## Performance Tips

1. **Build an index** for faster local searches:
   ```python
   mngs.scholar.build_index(recursive=True)
   ```

2. **Use caching** for repeated searches (enabled by default)

3. **Limit sources** when you know where to search:
   ```python
   papers = mngs.scholar.search_sync(
       query,
       web_sources=["arxiv"]  # Only search arXiv
   )
   ```

4. **Disable vector search** for faster results:
   ```python
   papers = mngs.scholar.search_sync(
       query,
       use_vector_search=False
   )
   ```

## Troubleshooting

### No PDF reader available
Install either PyMuPDF or PyPDF2:
```bash
pip install pymupdf
```

### Slow vector search
The first run downloads the embedding model. Subsequent runs will be faster.

### API rate limits
Some web sources have rate limits. The module handles this automatically with retries.

## Future Enhancements

- [ ] Additional web sources (Google Scholar, ResearchGate)
- [ ] Full-text search in PDFs
- [ ] Citation graph analysis
- [ ] Duplicate detection improvements
- [ ] Batch PDF processing
- [ ] Integration with reference managers

## Contributing

Contributions are welcome! Please see the main MNGS contributing guidelines.