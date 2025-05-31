# MNGS Naming Convention Guidelines

## Overview
This document defines the naming conventions for the MNGS (monogusa) Python package to ensure consistency, readability, and maintainability across the codebase.

## General Principles
1. **Clarity over brevity**: Names should be descriptive and self-documenting
2. **Consistency**: Similar concepts should follow similar naming patterns
3. **Python PEP 8 compliance**: Follow standard Python naming conventions
4. **Domain-appropriate**: Use terminology familiar to scientific Python users

## File Naming

### Python Files
- **Use snake_case**: `load_data.py`, `signal_processing.py`
- **Private modules**: Prefix with underscore: `_internal_utils.py`
- **No version suffixes**: Avoid `_v1.py`, `_dev.py`, `_working.py`
- **Descriptive names**: `save_image.py` not `si.py`

### Test Files
- **Mirror source structure**: `src/mngs/io/_load.py` → `tests/mngs/io/test__load.py`
- **Prefix with test_**: `test__load.py`, `test_comprehensive.py`
- **Double underscore for private**: `test__private_function.py`

## Code Naming Conventions

### Modules and Packages
```python
# Good
mngs.io
mngs.dsp
mngs.stats

# Bad
mngs.IO
mngs.DSP
mngs.Statistics
```

### Functions
```python
# Use snake_case for all functions
def load_data(filepath: str) -> Any:
    pass

def calculate_correlation(x: np.ndarray, y: np.ndarray) -> float:
    pass

# Private functions start with underscore
def _validate_input(data: Any) -> bool:
    pass
```

### Classes
```python
# Use PascalCase for classes
class DataLoader:
    pass

class SignalProcessor:
    pass

# Exception classes end with Error
class ValidationError(Exception):
    pass
```

### Variables
```python
# Use snake_case for variables
sample_rate = 1000
time_series = np.array([1, 2, 3])

# Constants in UPPER_SNAKE_CASE
DEFAULT_SAMPLE_RATE = 1000
MAX_ITERATIONS = 100

# Private variables start with underscore
_cache = {}
_internal_state = None
```

### Parameters
```python
# Be descriptive but concise
def filter_signal(
    data: np.ndarray,
    low_freq: float,    # Not 'lf' or 'low'
    high_freq: float,   # Not 'hf' or 'high'
    sample_rate: float, # Not 'sr' or 'fs'
    order: int = 4
) -> np.ndarray:
    pass

# Common parameter names to use consistently:
# - filepath (not filename, fname, path)
# - sample_rate (not sr, fs, sampling_rate)
# - n_samples (not num_samples, n_samp)
# - n_channels (not n_chs, num_channels)
```

## Module-Specific Conventions

### io Module
- Loading functions: `load_<format>()` (e.g., `load_json`, `load_pickle`)
- Saving functions: `save_<format>()` (e.g., `save_json`, `save_pickle`)
- Generic functions: `load()`, `save()` with format inference

### dsp Module
- Signal processing: `<operation>_signal()` (e.g., `filter_signal`, `resample_signal`)
- Frequency domain: `<operation>_spectrum()` (e.g., `compute_spectrum`)
- Time-frequency: `<operation>_tfr()` (e.g., `compute_tfr`)

### stats Module
- Statistical tests: `test_<name>()` (e.g., `test_normality`, `test_correlation`)
- Descriptive stats: `calculate_<metric>()` (e.g., `calculate_mean`, `calculate_variance`)
- Multiple testing: `correct_<method>()` (e.g., `correct_bonferroni`, `correct_fdr`)

### plt Module
- Plotting functions: `plot_<what>()` (e.g., `plot_time_series`, `plot_spectrum`)
- Axes operations: `set_<property>()` (e.g., `set_labels`, `set_title`)
- Figure operations: `save_figure()`, `close_figure()`

## Common Abbreviations
Use these consistently throughout the codebase:

| Full Term | Abbreviation | Usage |
|-----------|--------------|--------|
| array | arr | Variable names only |
| dataframe | df | Variable names only |
| figure | fig | Variable names only |
| axes | ax/axes | Variable names only |
| configuration | config | Variables and files |
| directory | dir | Parameters and variables |
| temporary | tmp | Variables and files |

## Anti-patterns to Avoid

### Don't use single letters (except in specific contexts)
```python
# Bad
def f(x, y):
    return x + y

# Good
def add_values(first: float, second: float) -> float:
    return first + second

# Exception: Mathematical contexts
def gaussian(x: float, mu: float, sigma: float) -> float:
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)
```

### Don't use type names in variable names
```python
# Bad
data_dict = {"a": 1}
values_list = [1, 2, 3]
name_str = "example"

# Good
data = {"a": 1}
values = [1, 2, 3]
name = "example"
```

### Don't use ambiguous abbreviations
```python
# Bad
proc_data()  # process? procedure? processor?
calc_res()   # result? residual? resolution?

# Good
process_data()
calculate_residuals()
```

## Migration Strategy
For existing code that doesn't follow these conventions:

1. **New code**: Must follow conventions
2. **Modified code**: Update to follow conventions
3. **Untouched code**: Update opportunistically
4. **Public APIs**: Maintain backward compatibility with deprecation warnings

Example deprecation:
```python
def ProcessData(*args, **kwargs):  # Old name
    warnings.warn(
        "ProcessData is deprecated, use process_data instead",
        DeprecationWarning,
        stacklevel=2
    )
    return process_data(*args, **kwargs)
```

## Enforcement
1. **Pre-commit hooks**: Automated checking for naming conventions
2. **Code reviews**: Manual verification of naming clarity
3. **Documentation**: All new code must include docstrings following conventions
4. **Tests**: Test function names must clearly indicate what is being tested

## Examples

### Good Module Structure
```python
# mngs/dsp/_filter.py
def bandpass_filter(
    data: np.ndarray,
    low_freq: float,
    high_freq: float,
    sample_rate: float,
    order: int = 4
) -> np.ndarray:
    """Apply bandpass filter to signal."""
    pass

def _design_butterworth_filter(
    low_freq: float,
    high_freq: float,
    sample_rate: float,
    order: int
) -> tuple:
    """Private function to design filter coefficients."""
    pass
```

### Good Test Structure
```python
# tests/mngs/dsp/test__filter.py
def test_bandpass_filter_removes_frequencies():
    """Test that bandpass filter removes out-of-band frequencies."""
    pass

def test_bandpass_filter_preserves_phase():
    """Test that zero-phase filtering preserves signal phase."""
    pass
```

## Revision History
- 2025-05-31: Initial version created
- Future updates will be documented here