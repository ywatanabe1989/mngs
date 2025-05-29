# MNGS Quick Start Guide (5 Minutes)

## Installation
```bash
pip install mngs
# or for development:
pip install -e /path/to/mngs_repo
```

## Basic Script Template

```python
#!/usr/bin/env python3
import sys
import matplotlib.pyplot as plt
import mngs

# Start (handles all initialization)
CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt)

# Your code here
data = mngs.io.load("./data.csv")
# ... process ...
mngs.io.save(results, "results.csv")

# Close (ensures proper cleanup)
mngs.gen.close(CONFIG)
```

## Essential Functions

### 1. File I/O
```python
# Load any file type
data = mngs.io.load("file.ext")  # Auto-detects format

# Save with auto directory creation
mngs.io.save(obj, "path/to/file.ext")
```

### 2. Configuration
```python
# Load all YAML configs from ./config/
CONFIG = mngs.io.load_configs()
print(CONFIG.model.batch_size)  # Dot access
```

### 3. Plotting
```python
# Enhanced matplotlib
fig, ax = mngs.plt.subplots(figsize=(6, 4))
ax.plot(x, y)
ax.set_xyt("Time (s)", "Voltage (mV)", "Signal")  # Set all labels at once
mngs.io.save(fig, "plot.png")  # Saves image + data
```

### 4. Data Processing
```python
# Pandas utilities
df_rounded = mngs.pd.round(df, 3)

# Signal processing
filtered = mngs.dsp.bandpass(signal, 1, 50, fs=1000)

# Statistics
result = mngs.stats.corr_test(x, y)
```

## Project Structure
```
project/
├── config/          # YAML configs
├── data/           # Input data
├── scripts/        # Your scripts
│   └── script_out/ # Auto-created outputs
├── examples/
└── tests/
```

## Common Patterns

### Pattern 1: Experiment Script
```python
import mngs

# Load config
CONFIG = mngs.io.load_configs()

# Start with seed for reproducibility
CONFIG, _, _, plt, _ = mngs.gen.start(
    sys, plt, seed=CONFIG.seed
)

# Run experiment
for epoch in range(CONFIG.n_epochs):
    # ... training code ...
    mngs.io.save(model, f"model_epoch_{epoch}.pth")

mngs.gen.close(CONFIG)
```

### Pattern 2: Data Analysis
```python
import mngs

# Quick analysis without full setup
df = mngs.io.load("data.csv")
df_clean = mngs.pd.round(df, 2)

# Plot results
fig, axes = mngs.plt.subplots(2, 2)
for ax, col in zip(axes.flat, df.columns):
    ax.hist(df[col])
    ax.set_xyt(col, "Count", f"Distribution of {col}")

mngs.io.save(fig, "distributions.png")
```

### Pattern 3: Signal Processing
```python
import mngs

# Load and process signal
signal = mngs.io.load("signal.npy")
fs = 1000  # sampling frequency

# Apply filters
filtered = mngs.dsp.bandpass(signal, 1, 50, fs)
hilbert = mngs.dsp.hilbert(filtered)

# Save results
mngs.io.save({
    "original": signal,
    "filtered": filtered,
    "hilbert": hilbert
}, "processed_signals.npz")
```

## Tips for Success

1. **Always use relative paths** starting with `./`
2. **Run scripts from project root**
3. **Use `mngs.gen.start()` for reproducibility**
4. **Let mngs handle directory creation**
5. **Check `./script_out/` for outputs**

## Next Steps

- Read the [Core Concepts](02_core_concepts.md) guide
- Explore [Module Overview](03_module_overview.md)
- Try the [Common Workflows](04_common_workflows.md)

## Help & Debugging

```python
# Check mngs version
import mngs
print(mngs.__version__)

# List available functions in a module
print(dir(mngs.io))

# Get help on any function
help(mngs.io.load)
```