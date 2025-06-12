# MNGS io Module - Detailed Reference

The `io` module provides a unified interface for file I/O operations, supporting 20+ file formats with automatic format detection and directory management.

## Core Philosophy

1. **Format agnostic**: Single interface for all file types
2. **Automatic organization**: Creates directories as needed
3. **Smart defaults**: Sensible options for each format
4. **Batch operations**: Handle multiple files easily
5. **Error resilience**: Graceful handling of issues

## Primary Functions

### mngs.io.save()

```python
mngs.io.save(
    obj,                    # Object to save
    path,                   # File path (can include new directories)
    makedirs=True,         # Create directories if needed
    verbose=False,         # Print save confirmation
    symlink=True,          # Create symlink in project root
    check=False,           # Verify save by loading
    **kwargs               # Format-specific options
)
```

**Supported formats and their options:**

#### NumPy Arrays (.npy, .npz)
```python
# Single array
mngs.io.save(array, "data.npy")

# Multiple arrays (compressed)
mngs.io.save({"x": x_array, "y": y_array}, "data.npz", compress=True)

# Memory-mapped array for large data
mngs.io.save(huge_array, "big.npy", mmap_mode='r+')
```

#### Pandas DataFrames (.csv, .xlsx)
```python
# CSV with options
mngs.io.save(df, "results.csv", index=False, encoding='utf-8')

# Excel with multiple sheets
mngs.io.save(
    {"sheet1": df1, "sheet2": df2}, 
    "report.xlsx",
    engine='openpyxl'
)
```

#### JSON/YAML (.json, .yaml, .yml)
```python
# Pretty-printed JSON
mngs.io.save(config, "config.json", indent=2, sort_keys=True)

# YAML with custom formatting
mngs.io.save(params, "params.yaml", default_flow_style=False)
```

#### Pickle (.pkl, .pickle)
```python
# Standard pickle
mngs.io.save(model, "model.pkl")

# Compressed pickle
mngs.io.save(large_object, "data.pkl.gz", compress=True)

# Specific protocol
mngs.io.save(obj, "data.pkl", protocol=4)
```

#### Matplotlib Figures (.png, .jpg, .pdf, .svg)
```python
# High-resolution PNG
mngs.io.save(fig, "plot.png", dpi=300, bbox_inches='tight')

# Vector format
mngs.io.save(fig, "plot.pdf", transparent=True)

# With data tracking (saves CSV too)
fig, ax = mngs.plt.subplots()
ax.plot(x, y)
mngs.io.save(fig, "analysis.png")  # Also saves analysis_data.csv
```

#### PyTorch Models (.pth, .pt)
```python
# Model state dict
mngs.io.save(model.state_dict(), "model.pth")

# Complete checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'loss': loss
}
mngs.io.save(checkpoint, "checkpoint.pth")
```

#### Text Files (.txt, .md)
```python
# String content
mngs.io.save("Results: accuracy = 0.95", "results.txt")

# List of lines
mngs.io.save(log_lines, "experiment.log", mode='a')  # Append mode

# With encoding
mngs.io.save(text, "doc.txt", encoding='utf-8')
```

#### HDF5 (.h5, .hdf5)
```python
# Nested dictionary structure
data = {
    'signals': {'raw': raw_data, 'filtered': filtered_data},
    'metadata': {'fs': 1000, 'duration': 60}
}
mngs.io.save(data, "experiment.h5", compression='gzip')
```

### mngs.io.load()

```python
loaded = mngs.io.load(
    path,                   # File path or glob pattern
    as_type=None,          # Force specific loader
    verbose=False,         # Print load confirmation
    **kwargs               # Format-specific options
)
```

**Format-specific loading options:**

#### Loading with options
```python
# CSV with specific columns
df = mngs.io.load("data.csv", usecols=['time', 'value'])

# JSON as OrderedDict
config = mngs.io.load("config.json", object_pairs_hook=OrderedDict)

# NPZ file - returns dict-like object
data = mngs.io.load("arrays.npz")
x = data['x']
y = data['y']

# Text file options
text = mngs.io.load("file.txt", as_lines=False)  # Full content
lines = mngs.io.load("file.txt", as_lines=True)  # List of lines
```

#### Glob pattern loading
```python
# Load multiple files
all_csvs = mngs.io.load("results/*.csv")  # Returns list

# Recursive glob
all_data = mngs.io.load("**/data.npy", recursive=True)
```

## Specialized Functions

### mngs.io.glob()

```python
files = mngs.io.glob(
    pattern,               # Glob pattern
    recursive=False,      # Search subdirectories
    sort=True,           # Sort results
    absolute=True        # Return absolute paths
)
```

**Examples:**
```python
# Find all numpy files
npy_files = mngs.io.glob("*.npy")

# Recursive search
all_csvs = mngs.io.glob("**/*.csv", recursive=True)

# Multiple patterns
files = mngs.io.glob(["*.png", "*.jpg", "*.pdf"])
```

### mngs.io.load_configs()

```python
CONFIG = mngs.io.load_configs(
    config_dir="./config",     # Directory with YAML files
    verbose=True,             # Print loaded files
    merge=True               # Merge all configs
)
```

**What it does:**
1. Finds all .yaml/.yml files in directory
2. Loads each file
3. Merges into single DotDict
4. Handles nested structures

**Example structure:**
```
config/
├── model.yaml      # model: {type: resnet, layers: 50}
├── training.yaml   # training: {lr: 0.001, epochs: 100}
└── data.yaml       # data: {path: ./data, batch_size: 32}

# Result:
CONFIG.model.type = "resnet"
CONFIG.training.lr = 0.001
CONFIG.data.batch_size = 32
```

### mngs.io.cache()

```python
@mngs.io.cache(cache_dir="./cache", ignore_args=['verbose'])
def expensive_computation(data, param1, param2, verbose=False):
    # This will be cached based on arguments
    return process(data, param1, param2)
```

## Utility Functions

### Path handling

```python
# Find latest file
latest = mngs.io.find_latest("results/*.csv")

# Check existence
if mngs.io.exists("data.npy"):
    data = mngs.io.load("data.npy")

# Get file info
info = mngs.io.info("large_file.h5")
# Returns: {'size': '1.5 GB', 'modified': '2025-05-31 10:30:00'}
```

### Batch operations

```python
# Save multiple files
data_dict = {'array1': arr1, 'array2': arr2, 'df': dataframe}
for name, obj in data_dict.items():
    mngs.io.save(obj, f"output/{name}")

# Load and process multiple files
for file in mngs.io.glob("raw/*.csv"):
    df = mngs.io.load(file)
    processed = process_dataframe(df)
    mngs.io.save(processed, file.replace('raw', 'processed'))
```

## Integration with gen module

The io module automatically uses paths from CONFIG when inside mngs.gen.start():

```python
CONFIG, *_ = mngs.gen.start(sys, plt, sdir="./results")

# These saves go to CONFIG.sdir/data/
mngs.io.save(results, "results.csv")      # ./results/timestamp/data/results.csv
mngs.io.save(model, "model.pkl")          # ./results/timestamp/data/model.pkl

# Plots go to CONFIG.sdir/fig/
mngs.io.save(figure, "analysis.png")      # ./results/timestamp/fig/analysis.png
```

## Best Practices

### 1. Let mngs handle paths

```python
# Bad: Manual path handling
os.makedirs("output/results/2025-05-31", exist_ok=True)
np.save("output/results/2025-05-31/data.npy", array)

# Good: Automatic path handling
mngs.io.save(array, "results/data.npy")  # Creates all directories
```

### 2. Use appropriate formats

```python
# Numerical data: NPY/NPZ (fast, compact)
mngs.io.save(large_array, "data.npy")

# Tabular data: CSV (human readable) or Parquet (efficient)
mngs.io.save(dataframe, "results.csv")  # For sharing
mngs.io.save(dataframe, "results.parquet")  # For large data

# Configuration: YAML (readable) or JSON (standard)
mngs.io.save(config, "config.yaml")  # Preferred for configs

# Complex objects: Pickle (Python only) or HDF5 (cross-platform)
mngs.io.save(model, "model.pkl")  # Python objects
mngs.io.save(nested_data, "data.h5")  # Complex hierarchies
```

### 3. Leverage format detection

```python
# Extension determines format automatically
mngs.io.save(data, "output.csv")   # Saves as CSV
mngs.io.save(data, "output.json")  # Saves as JSON
mngs.io.save(data, "output.pkl")   # Saves as pickle

# Force specific format if needed
mngs.io.save(data, "output.dat", as_type='npy')
```

### 4. Handle errors gracefully

```python
try:
    data = mngs.io.load("maybe_missing.csv")
except FileNotFoundError:
    print("File not found, using defaults")
    data = default_data

# Or check first
if mngs.io.exists("data.csv"):
    data = mngs.io.load("data.csv")
```

## Common Patterns

### Data pipeline

```python
# Load -> Process -> Save pipeline
raw_data = mngs.io.load("raw_data.csv")
processed = process_data(raw_data)
mngs.io.save(processed, "processed_data.csv")

# With verification
mngs.io.save(processed, "processed_data.csv", check=True)
```

### Incremental saving

```python
# Save with version numbers
for i, result in enumerate(results):
    mngs.io.save(result, f"results/iteration_{i:04d}.npy")

# Or with timestamps
timestamp = mngs.gen.gen_timestamp()
mngs.io.save(result, f"results_{timestamp}.pkl")
```

### Configuration management

```python
# Load all configs
CONFIG = mngs.io.load_configs("./config")

# Save current config
current_config = {
    'model': CONFIG.model,
    'training': CONFIG.training,
    'timestamp': mngs.gen.gen_timestamp()
}
mngs.io.save(current_config, "run_config.yaml")
```

### Format conversion

```python
# Convert between formats
data = mngs.io.load("data.csv")
mngs.io.save(data, "data.parquet")  # CSV -> Parquet
mngs.io.save(data, "data.xlsx")     # CSV -> Excel
mngs.io.save(data, "data.json")     # CSV -> JSON (if convertible)
```

## Performance Tips

### 1. Use compressed formats for large data

```python
# Compressed NPZ for multiple arrays
mngs.io.save({'data': huge_array}, "data.npz", compress=True)

# Compressed pickle
mngs.io.save(large_object, "object.pkl.gz")

# HDF5 with compression
mngs.io.save(nested_data, "data.h5", compression='gzip')
```

### 2. Memory-mapped arrays for huge data

```python
# Save for memory-mapped access
mngs.io.save(huge_array, "huge.npy")

# Load as memory-mapped
mmap_array = np.load("huge.npy", mmap_mode='r')
```

### 3. Parallel I/O for multiple files

```python
from multiprocessing import Pool

def process_and_save(file):
    data = mngs.io.load(file)
    result = heavy_processing(data)
    mngs.io.save(result, file.replace('input', 'output'))

# Process in parallel
files = mngs.io.glob("input/*.npy")
with Pool() as pool:
    pool.map(process_and_save, files)
```

## Troubleshooting

### Issue: "Permission denied"
```python
# Check file permissions
import os
os.chmod("file.pkl", 0o666)

# Or save to different location
mngs.io.save(data, "/tmp/temp_data.pkl")
```

### Issue: "Cannot determine format"
```python
# Explicitly specify format
mngs.io.save(data, "mydata.dat", as_type='npy')
data = mngs.io.load("mydata.dat", as_type='npy')
```

### Issue: "File too large"
```python
# Use chunked operations
# For DataFrames
for chunk in pd.read_csv("huge.csv", chunksize=10000):
    processed = process(chunk)
    mngs.io.save(processed, "output.csv", mode='a')

# For arrays
# Save in parts
np.save("part1.npy", array[:1000000])
np.save("part2.npy", array[1000000:])
```

## Summary

The io module abstracts away file format complexities, letting you focus on your data rather than I/O mechanics. With automatic format detection, directory creation, and integration with the broader mngs ecosystem, it provides a robust foundation for data management in scientific computing.