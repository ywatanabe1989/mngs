# [`mngs.general`](https://github.com/ywatanabe1989/mngs/tree/main/src/mngs/general/)

## Overview
The `mngs.general` module is a collection of utility functions and classes designed for general-purpose use in the MNGS project. It provides a wide range of tools to simplify common tasks and improve code efficiency across various domains.

## Installation
```bash
pip install mngs
```

## Features
- Miscellaneous utility functions
- Pandas DataFrame utilities
- Array dimension handling
- Caching and performance optimization
- Debugging tools
- Data processing and normalization
- File and system operations
- Reproducibility helpers

## Submodules
- `misc`: Miscellaneous utility functions
- `pandas_utils`: Utility functions for working with pandas DataFrames

## Quick Start
```python
from mngs.general import DotDict, TimeStamper, cache, close, dict_replace, embed

# Use DotDict for dot notation access to dictionary items
data = DotDict({'a': 1, 'b': {'c': 2}})
print(data.b.c)  # Output: 2

# Generate timestamps
ts = TimeStamper()
print(ts.stamp())  # Output: Current timestamp

# Use cache decorator for function memoization
@cache
def expensive_function(x):
    # Simulating an expensive operation
    return x ** 2

# Compare values
print(close(0.1 + 0.2, 0.3))  # Output: True

# Replace dictionary values
original = {'a': 1, 'b': 2, 'c': 3}
mapping = {1: 'one', 2: 'two', 3: 'three'}
print(dict_replace(original, mapping))  # Output: {'a': 'one', 'b': 'two', 'c': 'three'}

# Start an interactive debugging session
# embed()
```

## API Reference
### Classes
- `DimHandler`: Manages dimensions in arrays
- `DotDict`: Dictionary subclass allowing dot notation access
- `TimeStamper`: Generates and manages timestamps

### Decorators
- `cache`: Caches function results
- `deprecated`: Marks functions as deprecated
- `timeout`: Sets a timeout for function execution

### Context Managers
- `ci`: Changes the current working directory within a context

### Utility Functions
- `close`: Checks if two values are close to each other
- `dict_replace`: Replaces values in a dictionary based on a mapping
- `embed`: Starts an interactive Python session for debugging
- `less`: Displays content in a pager
- `mask_api`: Masks sensitive information in API responses
- `not_implemented`: Raises a NotImplementedError
- `paste`: Copies text to the clipboard
- `src`: Retrieves the source code of a function or class
- `wrap`: Wraps text to a specified width

### Data Processing
- `converters`: Conversion functions (e.g., to_numpy, to_torch)
- `norm`: Normalization functions
- `symlog`: Symmetric log transformation
- `transpose`: Transposes data structures

### File and System Operations
- `email`: Functions for sending emails and notifications
- `shell`: Functions for running shell commands and scripts
- `tee`: Redirects output to both stdout and a file
- `title2path`: Converts a title to a valid file path

### Reproducibility
- `reproduce`: Functions for ensuring reproducibility (e.g., fix_seeds, gen_ID)

## Use Cases
- Simplifying common programming tasks
- Enhancing code readability and maintainability
- Debugging and development support
- Data manipulation and processing
- System interaction and automation
- Ensuring reproducibility in scientific computing

## Contributing
Contributions to improve `mngs.general` are welcome. Please submit pull requests or open issues on the GitHub repository.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
Yusuke Watanabe (ywata1989@gmail.com)

For more information and updates, please visit the [mngs GitHub repository](https://github.com/ywatanabe1989/mngs).
