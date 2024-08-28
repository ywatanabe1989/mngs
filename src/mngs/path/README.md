# [`mngs.path`](https://github.com/ywatanabe1989/mngs/tree/main/src/mngs/path/)

## Overview
The `mngs.path` module provides a set of utilities for handling file paths, directory operations, and version control related tasks in Python. It simplifies common path-related operations and adds functionality for finding files and directories.

## Installation
```bash
$ pip install mngs
```

## Features
- Path manipulation and information retrieval
- Directory and file search functionality
- Git repository root finding
- Version control helpers

## Quick Start
```python
import mngs

# Path information
fpath = mngs.path.this_path()  # Returns the current file path, e.g., "/tmp/fake.py"
spath = mngs.path.spath()  # Returns a safe path, e.g., '/tmp/fake-ywatanabe/.'
dir, fname, ext = mngs.path.split(fpath)  # Splits path into directory, filename, and extension

# Find directories and files
mngs.path.find_dir(".", "path")  # Finds directories matching the pattern, e.g., [./src/mngs/path]
mngs.path.find_file(".", "*wavelet.py")  # Finds files matching the pattern, e.g., ['./src/mngs/dsp/_wavelet.py']

# Git and versioning
git_root = mngs.path.find_git_root()  # Finds the root of the Git repository
# mngs.path.find_latest()  # Finds the latest version of a file (commented out in the example)
# mngs.path.increment_version()  # Increments the version of a file (commented out in the example)
```

## API Reference
- `mngs.path.this_path()`: Returns the path of the current file
- `mngs.path.spath()`: Returns a safe path (user-specific temporary directory)
- `mngs.path.split(path)`: Splits a path into directory, filename, and extension
- `mngs.path.find_dir(root, pattern)`: Finds directories matching a pattern
- `mngs.path.find_file(root, pattern)`: Finds files matching a pattern
- `mngs.path.find_git_root()`: Finds the root directory of the current Git repository
- `mngs.path.find_latest()`: Finds the latest version of a file (implementation details needed)
- `mngs.path.increment_version()`: Increments the version of a file (implementation details needed)

## Use Cases
- Simplifying path manipulations in Python scripts
- Searching for specific files or directories within a project
- Managing versioned files and directories
- Working with Git repositories programmatically

## Contact
Yusuke Watanabe (ywata1989@gmail.com)

For more information and updates, please visit the [mngs GitHub repository](https://github.com/ywatanabe1989/mngs).
