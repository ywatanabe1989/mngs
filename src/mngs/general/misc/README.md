# Misc Submodule

The Misc submodule is a collection of miscellaneous utility functions that provide a wide range of functionality for various tasks in the MNGS project.

## Installation

This submodule is part of the MNGS project. To use it, ensure you have the MNGS package installed:

```
pip install mngs
```

## Usage

To use functions from this submodule, import them as follows:

```python
from mngs.general.misc import function_name

# Example usage
result = function_name(arguments)
```

## Available Functions

### File and Directory Operations
- `ci`: Context manager for changing the current working directory.
- `search`: Search for files or directories.
- `natglob`: Natural sort for glob results.
- `tee`: Redirect output to both stdout and a file.

### Data Manipulation
- `close`: Check if two values are close to each other.
- `dict_replace`: Replace values in a dictionary based on a mapping.
- `DimHandler`: Class for handling dimensions in arrays.
- `merge_dicts_wo_overlaps`: Merge dictionaries without overlapping keys.
- `mv_col`: Move a column in a pandas DataFrame.
- `pop_keys`: Remove and return multiple keys from a dictionary.
- `uq`: Get unique elements from an iterable.

### Text Processing
- `wrap`: Wrap text to a specified width.
- `color_text`: Add color to text output.
- `ct`: Alias for color_text.
- `connect_nums`: Connect a list of numbers into a string.
- `grep`: Search for a pattern in a file or string.
- `print_block`: Print text in a formatted block.

### Development Tools
- `deprecated`: Decorator to mark functions as deprecated.
- `embed`: Start an interactive Python session.
- `less`: Display content in a pager.
- `mask_api`: Mask sensitive information in API responses.
- `not_implemented`: Raise a NotImplementedError with a custom message.
- `paste`: Copy text to the clipboard.
- `src`: Get the source code of a function or class.
- `timeout`: Set a timeout for a function execution.
- `describe`: Describe an object's attributes and methods.

### Notifications and Emails
- `notify`: Send a notification.
- `send_gmail`: Send an email using Gmail.

### Reproducibility and Identifiers
- `fix_seeds`: Set random seeds for reproducibility.
- `gen_ID`: Generate a unique identifier.
- `start`: Initialize and start a process or task.

### LaTeX Utilities
- `add_hat_in_the_latex_style`: Add a hat to a LaTeX string.
- `to_the_latex_style`: Convert a string to LaTeX style.

### Miscellaneous
- `is_defined_global`: Check if a variable is defined in the global scope.
- `is_defined_local`: Check if a variable is defined in the local scope.
- `is_listed_X`: Check if an object is a list-like container.
- `listed_dict`: Create a dictionary with list values.
- `partial_at`: Create a partial function with positional arguments.
- `readable_bytes`: Convert bytes to a human-readable format.
- `wait_key`: Wait for a keypress.

Refer to the individual function docstrings for detailed information on usage, parameters, and return values.

## Contributing

Contributions to the Misc submodule are welcome. Please ensure that any new functions are well-documented and include appropriate unit tests.

## License

This submodule is part of the MNGS project and is subject to its licensing terms.
