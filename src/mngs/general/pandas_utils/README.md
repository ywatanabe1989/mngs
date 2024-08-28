# Pandas Utils Submodule

The Pandas Utils submodule provides a collection of utility functions for working with pandas DataFrames, enhancing data manipulation and analysis capabilities in the MNGS project.

## Installation

This submodule is part of the MNGS project. To use it, ensure you have the MNGS package installed:

```
pip install mngs
```

## Usage

To use functions from this submodule, import them as follows:

```python
from mngs.general.pandas_utils import function_name

# Example usage
result = function_name(dataframe, arguments)
```

## Available Functions

### DataFrame Column Manipulation
- `col_to_last(df, col)`: Move a column to the last position in a DataFrame.
- `col_to_top(df, col)`: Move a column to the first position in a DataFrame.
- `merge_columns(df, columns, new_column_name, sep=' ')`: Merge multiple columns into a single column.

### DataFrame Conversion and Validation
- `force_dataframe(data)`: Ensure that the input is a pandas DataFrame.

### Warning Management
- `ignore_SettingWithCopyWarning()`: Context manager to ignore SettingWithCopyWarning.

## Detailed Function Descriptions

### col_to_last(df, col)
Moves the specified column to the last position in the DataFrame.

- Parameters:
  - df (pandas.DataFrame): The input DataFrame.
  - col (str): The name of the column to move.
- Returns:
  - pandas.DataFrame: The DataFrame with the specified column moved to the last position.

### col_to_top(df, col)
Moves the specified column to the first position in the DataFrame.

- Parameters:
  - df (pandas.DataFrame): The input DataFrame.
  - col (str): The name of the column to move.
- Returns:
  - pandas.DataFrame: The DataFrame with the specified column moved to the first position.

### merge_columns(df, columns, new_column_name, sep=' ')
Merges multiple columns into a single column.

- Parameters:
  - df (pandas.DataFrame): The input DataFrame.
  - columns (list): List of column names to merge.
  - new_column_name (str): Name of the new merged column.
  - sep (str, optional): Separator to use between merged values. Defaults to a space.
- Returns:
  - pandas.DataFrame: The DataFrame with the new merged column.

### force_dataframe(data)
Ensures that the input is a pandas DataFrame. If not, attempts to convert it.

- Parameters:
  - data: Input data to convert to a DataFrame.
- Returns:
  - pandas.DataFrame: The input data as a DataFrame.
- Raises:
  - ValueError: If the input cannot be converted to a DataFrame.

### ignore_SettingWithCopyWarning()
Context manager to temporarily ignore pandas SettingWithCopyWarning.

- Usage:
  ```python
  with ignore_SettingWithCopyWarning():
      # Your code that may raise SettingWithCopyWarning
  ```

Refer to the individual function docstrings for more detailed information on usage and parameters.

## Contributing

Contributions to the Pandas Utils submodule are welcome. Please ensure that any new functions are well-documented and include appropriate unit tests.

## License

This submodule is part of the MNGS project and is subject to its licensing terms.
