<!-- ---
!-- title: ./mngs_repo/src/mngs/decorators/README.md
!-- author: ywatanabe
!-- date: 2024-11-25 13:37:09
!-- --- -->


# Decorators

# Decorators must be able to combine in any order

## batch_fn
A decorator for processing data in batches.

### Features
- Requires explicit `batch_size` keyword argument
  - Automatically applies `batch_size=-1` if not specified
- Supports multiple batch dimensions: 
  - Single dimension: `batch_size=4`
  - Multiple dimensions: `batch_size=(4, 8)`
- Guarantees consistent output regardless of batch size
- Supports NumPy arrays, PyTorch tensors, Pandas DataFrames

## torch_fn
A decorator for PyTorch function compatibility.

### Features
- Handles nested torch_fn decorators
- Automatically converts `axis=X` to `dim=X` for torch functions
- Automatically applies `device="cuda"` if available
- Preserves input data types in output:
  - NumPy arrays → NumPy arrays
  - Pandas objects → Pandas objects
  - Xarray objects → Xarray objects

## numpy_fn
A decorator for NumPy function compatibility.

### Features
- Automatically converts torch tensors to numpy arrays
- Preserves input data types in output
- Handles axis-related parameter conversions

## pandas_fn
A decorator for Pandas function compatibility.

### Features
- Automatically converts input data to pandas objects
- Preserves index and column information
- Handles DataFrame and Series operations consistently

## xarray_fn
A decorator for Xarray function compatibility.

### Features
- Automatically converts input data to xarray objects
- Preserves coordinate and dimension information
- Supports labeled dimension operations
