# General Module

The General Module is a collection of utility functions and classes designed for general-purpose use in the MNGS project. It provides a wide range of tools to simplify common tasks and improve code efficiency.

## Installation

This module is part of the MNGS project. To use it, ensure you have the MNGS package installed:

```
pip install mngs
```

## Submodules

### misc
Contains miscellaneous utility functions for various tasks. See [misc/README.md](misc/README.md) for a detailed list of functions and their usage.

### pandas_utils
Provides utility functions for working with pandas DataFrames. See [pandas_utils/README.md](pandas_utils/README.md) for more information on available functions.

## Main Components

### Classes
- `DimHandler`: Manages dimensions in arrays, simplifying shape manipulations.
- `DotDict`: A dictionary subclass allowing dot notation access to items.
- `TimeStamper`: Generates and manages timestamps for logging and tracking purposes.

### Decorators
- `cache`: Caches function results for improved performance.
- `deprecated`: Marks functions as deprecated, issuing warnings when used.
- `timeout`: Sets a timeout for function execution to prevent infinite loops.

### Context Managers
- `ci`: Changes the current working directory within a context.

### Utility Functions
- `close`: Checks if two values are close to each other.
- `dict_replace`: Replaces values in a dictionary based on a mapping.
- `embed`: Starts an interactive Python session for debugging.
- `less`: Displays content in a pager for easy viewing.
- `mask_api`: Masks sensitive information in API responses.
- `not_implemented`: Raises a NotImplementedError with a custom message.
- `paste`: Copies text to the clipboard.
- `src`: Retrieves the source code of a function or class.
- `wrap`: Wraps text to a specified width for formatting.

### Data Processing
- `converters`: Module with various conversion functions (e.g., to_numpy, to_torch).
- `norm`: Module with normalization functions.
- `symlog`: Performs symmetric log transformation.
- `transpose`: Transposes data structures.

### File and System Operations
- `email`: Functions for sending emails and notifications.
- `shell`: Functions for running shell commands and scripts.
- `tee`: Redirects output to both stdout and a file.
- `title2path`: Converts a title to a valid file path.

### Reproducibility
- `reproduce`: Functions for ensuring reproducibility (e.g., fix_seeds, gen_ID).

### Miscellaneous
- `start`: Initializes and starts a process or task.
- `dict2str`: Converts a dictionary to a string representation.
- `latex`: Functions for LaTeX-related operations.

## Usage

To use functions from this module, import them as follows:

```python
from mngs.general import function_name

# Example usage
result = function_name(arguments)
```

Refer to the individual function and class docstrings for detailed information on usage and parameters.

## Contributing

Contributions to the General Module are welcome. Please ensure that any new functions or classes are well-documented and include appropriate unit tests.

## License

This module is part of the MNGS project and is subject to its licensing terms.

