# MNGS IO Module

This module provides various input/output operations for the MNGS package.

## Functions

### glob

`glob(expression)`

Perform a glob operation with natural sorting and extended pattern support.

### load

`load(lpath, show=False, verbose=False, **kwargs)`

Load data from various file formats.

### load_configs

`load_configs(IS_DEBUG=None, show=False, verbose=False)`

Load and process configuration files from the ./config directory.

## Usage

```python
from mngs.io import glob, load, load_configs

# Use glob to find files
files = glob('data/*.txt')

# Load a file
data = load('data/example.csv')

# Load configurations
configs = load_configs()
```

For more detailed information on each function, please refer to their respective docstrings.
