# IO Consistency Testing Framework

## Overview

This document provides an overview of the IO consistency testing framework implemented for the MNGS repository. These tests are designed to verify data integrity and consistency across save/load operations for various data types, formats, and structures.

## Test Categories

The IO consistency tests are organized into the following categories:

### 1. Basic IO Consistency Tests (`test_io_consistency.py`)

- Round-trip verification for basic data types (arrays, matrices, lists)
- Tests for basic data structures including dictionaries, lists, and tensors
- Multiple save/load cycle tests to ensure idempotent operations
- Basic DataFrame serialization and deserialization

### 2. DataFrame-Specific Tests (`test_dataframe_consistency.py`)

- Index preservation (standard indices and named indices)
- Multi-index preservation
- Column order preservation
- Multi-index columns preservation
- Named columns preservation
- Empty DataFrame structure preservation
- Multi-format round-trip testing specific to DataFrames

### 3. Special Data Types Tests (`test_special_datatypes_consistency.py`)

- Datetime values preservation
- Timezone-aware datetime preservation
- Categorical data type preservation
- Nullable integer types (Int64) preservation
- Complex number handling
- Boolean data type preservation
- NaN and None values handling

### 4. Nested Structure Tests (`test_nested_structures_consistency.py`)

- Dictionaries of DataFrames
- Lists of DataFrames
- Deeply nested dictionary structures
- DataFrames with nested objects (lists/dicts in columns)
- NumPy structured arrays
- NPZ dictionary preservation

### 5. Cross-Format Compatibility Tests (`test_cross_format_compatibility.py`)

- Numeric data across different formats (NPY, CSV, PKL, JSON)
- DataFrame compatibility across formats
- Dictionary serialization across formats (PKL, JSON, YAML)
- NumPy to PyTorch conversions
- Pandas to NumPy conversions
- List conversion compatibility
- Multi-format round-trip conversions

## Usage

These tests can be run individually or as a complete suite:

```bash
# Run all IO consistency tests
./run_tests.sh tests/mngs/io/test_io_consistency.py tests/mngs/io/test_dataframe_consistency.py tests/mngs/io/test_special_datatypes_consistency.py tests/mngs/io/test_nested_structures_consistency.py tests/mngs/io/test_cross_format_compatibility.py

# Run a specific category of tests
./run_tests.sh tests/mngs/io/test_dataframe_consistency.py
```

## Implementation Details

1. **Test Structure**: Each test follows a similar pattern:
   - Create test data
   - Save to one or more formats
   - Load the data back
   - Verify integrity with appropriate assertions

2. **Temporary Files**: All tests use `tempfile.mkdtemp()` to create temporary directories and properly clean up after testing.

3. **Assertions**: Tests use appropriate assertion methods based on data type:
   - `np.testing.assert_array_equal()` for NumPy arrays
   - `pd.testing.assert_frame_equal()` for Pandas DataFrames
   - `torch.all()` for PyTorch tensors
   - Standard `assert` for primitive types and collections

4. **Format Support**: Tests cover most supported formats:
   - NumPy: `.npy`, `.npz`
   - Pandas: `.csv`
   - PyTorch: `.pt`, `.pth`
   - Serialization: `.pkl`, `.json`, `.yaml`

## Key Focus Areas

1. **Data Integrity**: Ensuring values are preserved exactly during serialization and deserialization.
2. **Structure Preservation**: Verifying that complex data structures maintain their organization.
3. **Metadata Retention**: Testing that metadata like DataFrame indices, column names, etc. are preserved.
4. **Type Handling**: Accounting for type conversions when they occur (e.g., CSV serialization).
5. **Cross-Format Support**: Testing interoperability between different file formats.

## Known Limitations

1. Some format conversions may lose type information (particularly CSV).
2. NPZ files, when loaded, return arrays as a list rather than with their original keys.
3. Some special data types (categorical, etc.) are only preserved in specific formats (primarily pickle).

## Future Improvements

1. Add tests for additional specialized types (sparse matrices, etc.)
2. Expand cross-format compatibility testing
3. Add more thorough edge case handling (extremely large files, unusual data values)
4. Implement tests for concurrent read/write operations

---

For detailed implementation, refer to the individual test files in the `tests/mngs/io/` directory.