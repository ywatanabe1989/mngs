# mngs.gen Module Documentation

## Overview

The `mngs.gen` module provides general utilities for managing experiment environments, logging, and system configurations in scientific Python projects. It's the backbone of the mngs framework, handling initialization, cleanup, and various utility functions.

## Core Functions

### Environment Management

#### `mngs.gen.start()`
Initialize experiment environment with reproducible settings.

```python
import sys
import matplotlib.pyplot as plt
import mngs

# Basic usage
CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt)

# With custom settings
CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
    sys, plt,
    seed=42,                    # Random seed for reproducibility
    sdir_suffix="experiment1",  # Custom save directory suffix
    verbose=True,              # Print detailed information
    agg=True,                  # Use matplotlib Agg backend
    fig_size_mm=(160, 100),    # Figure size in millimeters
    dpi_save=300,              # DPI for saving figures
)
```

**What it does:**
- Creates unique experiment ID (e.g., "GlBZ")
- Sets up logging directory structure
- Redirects stdout/stderr to log files
- Fixes random seeds for reproducibility
- Configures matplotlib settings
- Returns configuration dictionary with all settings

**Returns:**
- `CONFIG`: DotDict containing all configuration parameters
- `sys.stdout`, `sys.stderr`: Redirected output streams
- `plt`: Configured matplotlib.pyplot module
- `CC`: Color cycle dictionary for consistent plotting

#### `mngs.gen.close()`
Clean up experiment environment and save logs.

```python
# Basic usage
mngs.gen.close(CONFIG)

# With notification
mngs.gen.close(CONFIG, notify=True, message="Experiment completed!")

# With exit status
mngs.gen.close(CONFIG, exit_status=0)  # Success
mngs.gen.close(CONFIG, exit_status=1)  # Error
```

**What it does:**
- Saves final configuration to YAML and pickle files
- Moves logs from RUNNING/ to FINISHED_SUCCESS/ or FINISHED_ERROR/
- Removes ANSI escape codes from log files
- Optionally sends notification with results
- Calculates and logs total runtime

### Utility Functions

#### `mngs.gen.gen_ID()`
Generate unique identifier strings.

```python
# Generate 4-character ID (default)
id = mngs.gen.gen_ID()  # e.g., "Ab3D"

# Generate custom length ID
id = mngs.gen.gen_ID(N=8)  # e.g., "Ab3Df9Kl"
```

#### `mngs.gen.fix_seeds()`
Fix random seeds across multiple libraries for reproducibility.

```python
import os, random, numpy as np, torch

mngs.gen.fix_seeds(
    os=os,
    random=random,
    np=np,
    torch=torch,
    seed=42,
    verbose=True
)
```

#### `mngs.gen.TimeStamper`
Class for generating consistent timestamps.

```python
ts = mngs.gen.TimeStamper()

# Get formatted timestamp
timestamp = ts.get_timestamp()  # "2024-03-15 14:30:45"

# Get timestamp for filenames
file_timestamp = ts.get_file_timestamp()  # "2024Y-03M-15D-14h30m45s"
```

#### `mngs.gen.DimHandler`
Utility for handling array dimensions.

```python
dh = mngs.gen.DimHandler()

# Ensure array has specific number of dimensions
arr_3d = dh.ensure_3d(arr_2d)  # Adds dimension if needed

# Get dimension information
n_dims = dh.get_n_dims(array)
shape_info = dh.get_shape_info(array)
```

### Path Utilities

#### `mngs.gen.title2path()`
Convert title strings to valid file paths.

```python
title = "My Experiment: Results (v2.0)"
path = mngs.gen.title2path(title)  # "my_experiment_results_v2_0"
```

#### `mngs.gen.symlink()`
Create symbolic links with safety checks.

```python
mngs.gen.symlink(
    source="/path/to/source",
    target="/path/to/link",
    overwrite=True
)
```

### Data Transformation

#### `mngs.gen.to_even()` / `mngs.gen.to_odd()`
Convert numbers to nearest even/odd values.

```python
even = mngs.gen.to_even(3)   # 4
odd = mngs.gen.to_odd(4)     # 5
```

#### `mngs.gen.to_rank()`
Convert values to ranks.

```python
values = [3.2, 1.5, 4.8, 2.1]
ranks = mngs.gen.to_rank(values)  # [3, 1, 4, 2]
```

#### `mngs.gen.transpose()`
Transpose nested lists or arrays.

```python
data = [[1, 2, 3], [4, 5, 6]]
transposed = mngs.gen.transpose(data)  # [[1, 4], [2, 5], [3, 6]]
```

### System Information

#### `mngs.gen.print_config()`
Print system configuration information.

```python
mngs.gen.print_config()
# Outputs: Python version, installed packages, system info
```

#### `mngs.gen.list_packages()`
List installed packages with versions.

```python
packages = mngs.gen.list_packages()
# Returns dict: {"numpy": "1.21.0", "pandas": "1.3.0", ...}
```

#### `mngs.gen.check_host()`
Check if running on specific host.

```python
if mngs.gen.check_host("gpu-server"):
    # Use GPU-specific settings
    batch_size = 64
else:
    batch_size = 16
```

### Interactive Utilities

#### `mngs.gen.tee()`
Redirect output to both console and file.

```python
import sys
sys.stdout, sys.stderr = mngs.gen.tee(sys, sdir="./logs/")
```

#### `mngs.gen.less()`
Display content with pagination (like Unix less command).

```python
long_text = "Very long content...\n" * 100
mngs.gen.less(long_text)
```

#### `mngs.gen.embed()`
Launch IPython debugger at current point.

```python
# For debugging
mngs.gen.embed()  # Drops into IPython shell
```

## Directory Structure

When using `mngs.gen.start()`, the following directory structure is created:

```
script_out/
└── RUNNING/
    └── {ID}/
        ├── CONFIGS/
        │   ├── CONFIG.pkl
        │   └── CONFIG.yaml
        └── logs/
            ├── stdout.log
            └── stderr.log
```

After `mngs.gen.close()`:

```
script_out/
└── FINISHED_SUCCESS/  # or FINISHED_ERROR/
    └── {ID}/
        ├── CONFIGS/
        │   ├── CONFIG.pkl
        │   └── CONFIG.yaml
        └── logs/
            ├── stdout.log
            └── stderr.log
```

## Best Practices

### 1. Always Use start/close Pair
```python
import sys
import matplotlib.pyplot as plt
import mngs

def main():
    # Your experiment code here
    pass

if __name__ == "__main__":
    # Initialize
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt)
    
    try:
        # Run experiment
        exit_status = main()
    except Exception as e:
        print(f"Error: {e}")
        exit_status = 1
    finally:
        # Always close
        mngs.gen.close(CONFIG, exit_status=exit_status)
```

### 2. Use Unique IDs for Tracking
```python
CONFIG, *_ = mngs.gen.start(sys, plt)
experiment_id = CONFIG.ID  # Use this for tracking

# Save results with ID
mngs.io.save(results, f"./results_{experiment_id}.pkl")
```

### 3. Leverage Color Cycle
```python
CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt)

# Use consistent colors across plots
plt.plot(x, y1, color=CC["blue"], label="Method 1")
plt.plot(x, y2, color=CC["red"], label="Method 2")
```

### 4. Custom Save Directories
```python
# For different experiment types
CONFIG, *_ = mngs.gen.start(
    sys, plt,
    sdir_suffix="ablation_study"
)

# For debugging (won't clutter main results)
CONFIG, *_ = mngs.gen.start(
    sys, plt,
    sdir="/tmp/debug_runs/"
)
```

## Common Use Cases

### 1. Scientific Experiment
```python
import sys
import numpy as np
import matplotlib.pyplot as plt
import mngs

def run_experiment(config):
    # Load data
    data = mngs.io.load("./data/dataset.npy")
    
    # Process
    results = process_data(data, config)
    
    # Save results
    mngs.io.save(results, "./results.pkl")
    
    # Plot
    fig, ax = plt.subplots()
    ax.plot(results)
    mngs.io.save(fig, "./results_plot.png")
    
    return 0  # Success

if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt,
        seed=42,  # Reproducibility
        agg=True  # Headless plotting
    )
    
    exit_status = run_experiment(CONFIG)
    
    mngs.gen.close(CONFIG, exit_status=exit_status)
```

### 2. Parameter Sweep
```python
import sys
import matplotlib.pyplot as plt
import mngs

parameters = [0.1, 0.5, 1.0, 2.0]

for param in parameters:
    # Each run gets unique ID
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt,
        sdir_suffix=f"param_{param}"
    )
    
    try:
        results = run_with_param(param)
        mngs.io.save(results, f"./results_param_{param}.pkl")
        exit_status = 0
    except Exception as e:
        print(f"Failed with param {param}: {e}")
        exit_status = 1
    finally:
        mngs.gen.close(CONFIG, exit_status=exit_status)
```

### 3. Debug Mode
```python
# Create ./config/IS_DEBUG.yaml with content: IS_DEBUG: true

CONFIG, *_ = mngs.gen.start(sys, plt)
if "DEBUG" in CONFIG.ID:
    print("Running in debug mode")
    # Use smaller dataset, fewer iterations, etc.
```

## Troubleshooting

### Issue: Logs not saving
```python
# Ensure proper cleanup
import atexit

CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt)
atexit.register(lambda: mngs.gen.close(CONFIG))
```

### Issue: Multiple start calls
```python
# Wrong
for i in range(10):
    CONFIG, *_ = mngs.gen.start(sys, plt)  # Creates new logs each time
    
# Right
CONFIG, *_ = mngs.gen.start(sys, plt)
for i in range(10):
    # Use same CONFIG throughout
    pass
mngs.gen.close(CONFIG)
```

### Issue: Can't find logs
```python
CONFIG, *_ = mngs.gen.start(sys, plt, verbose=True)
print(f"Logs will be saved to: {CONFIG.SDIR}")
```

## See Also

- [mngs.io](../io/README.md) - File I/O operations
- [mngs.plt](../plt/README.md) - Enhanced plotting utilities
- [Agent Guidelines](../../agent_guidelines/02_core_concepts.md) - Core concepts