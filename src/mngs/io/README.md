# [`mngs.io`](https://github.com/ywatanabe1989/mngs/tree/main/src/mngs/io/)
Python input/output utilities.

## Overview
The `mngs.io` module provides a set of convenient functions for handling various input/output operations in Python. It simplifies the process of saving and loading different data types and file formats.

## Installation
```bash
pip install mngs
```

## Features
- Unified interface for saving and loading various file formats
- Support for common data types: NumPy arrays, Pandas DataFrames, PyTorch tensors
- Handling of serializable Python objects (dict, list, etc.)
- Support for image saving (PNG, TIFF)
- YAML and JSON file handling
- Caching functionality for improved performance

## Quick Start
```python
import mngs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# NumPy array (.npy)
arr = np.array([1, 2, 3])
mngs.io.save(arr, "xxx.npy")
arr = mngs.io.load("xxx.npy")

# Pandas DataFrame (.csv)
df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
mngs.io.save(df, "xxx.csv")
df = mngs.io.load("xxx.csv")

# PyTorch tensor (.pth)
tensor = torch.tensor([1, 2, 3])
mngs.io.save(tensor, "xxx.pth")
tensor = mngs.io.load("xxx.pth")

# Serializable object (.pkl)
_dict = {"a": 1, "b": 2, "c": [3, 4, 5]}
mngs.io.save(_dict, "xxx.pkl")
_dict = mngs.io.load("xxx.pkl")

# Matplotlib figure (.png or .tiff)
plt.figure()
plt.plot(np.array([1, 2, 3]))
mngs.io.save(plt, "xxx.png")  # or "xxx.tiff"

# YAML file
mngs.io.save(_dict, "xxx.yaml")
_dict = mngs.io.load("xxx.yaml")

# JSON file
mngs.io.save(_dict, "xxx.json")
_dict = mngs.io.load("xxx.json")

# Using cache
@mngs.io.cache
def expensive_function(x):
    # Some time-consuming computation
    return x * 2

result = expensive_function(5)  # First call, computes and caches
result = expensive_function(5)  # Second call, retrieves from cache
```

## API Reference
- `mngs.io.save(obj, path)`: Save an object to a file
- `mngs.io.load(path)`: Load an object from a file
- `mngs.io.cache`: Decorator for caching function results
- `mngs.io.glob(pattern)`: Find files matching a pattern
- `mngs.io.reload(module)`: Reload a Python module

The `save` and `load` functions automatically detect the file format based on the file extension and handle the data accordingly.

## Supported File Formats
- `.npy`: NumPy arrays
- `.csv`: Pandas DataFrames
- `.pth`: PyTorch tensors
- `.pkl`: Serializable Python objects
- `.png`, `.tiff`: Matplotlib figures
- `.yaml`: YAML files
- `.json`: JSON files

## Contributing
Contributions to improve `mngs.io` are welcome. Please submit pull requests or open issues on the GitHub repository.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
Yusuke Watanabe (ywata1989@gmail.com)

For more information and updates, please visit the [mngs GitHub repository](https://github.com/ywatanabe1989/mngs).

