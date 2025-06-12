# MNGS Complete Reference Guide

**Version**: 1.0  
**Last Updated**: 2025-05-31  
**Test Coverage**: 100% (core modules)

This document provides a complete reference for all functions, classes, and methods in the MNGS framework.

## Table of Contents

1. [gen - Environment & Experiment Management](#gen-module)
2. [io - File Input/Output](#io-module)
3. [plt - Enhanced Plotting](#plt-module)
4. [pd - Pandas Utilities](#pd-module)
5. [dsp - Digital Signal Processing](#dsp-module)
6. [stats - Statistical Analysis](#stats-module)
7. [ai - Machine Learning & AI](#ai-module)
8. [nn - Neural Network Layers](#nn-module)
9. [str - String Utilities](#str-module)
10. [path - Path Utilities](#path-module)
11. [decorators - Function Decorators](#decorators-module)
12. [Other Utilities](#other-utilities)

---

## gen Module

### Core Functions

#### `mngs.gen.start(sys, plt, sdir="./", verbose=True, random_seed=None, **kwargs)`
Initialize experiment environment with automatic logging and configuration.

**Parameters:**
- `sys`: System module
- `plt`: Matplotlib.pyplot module
- `sdir`: Save directory (default: "./")
- `verbose`: Enable verbose output (default: True)
- `random_seed`: Random seed for reproducibility (default: None)
- `**kwargs`: Additional configuration options

**Returns:**
- `CONFIG`: Configuration object with experiment settings
- `sys.stdout`: Redirected stdout for logging
- `sys.stderr`: Redirected stderr for logging
- `plt`: Configured matplotlib module
- `CC`: Closed captioning object for logging

**Example:**
```python
import sys
import matplotlib.pyplot as plt
import mngs

CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
    sys, plt, 
    sdir="./output", 
    random_seed=42
)
```

#### `mngs.gen.close(CONFIG)`
Clean up experiment environment and save logs.

**Parameters:**
- `CONFIG`: Configuration object from start()

**Example:**
```python
mngs.gen.close(CONFIG)
```

### Utility Functions

#### `mngs.gen.title2path(title)`
Convert a title string to a valid file path.

**Parameters:**
- `title`: String to convert

**Returns:**
- Safe filename string

#### `mngs.gen.gen_ID(n=8)`
Generate a unique identifier.

**Parameters:**
- `n`: Length of ID (default: 8)

**Returns:**
- Unique ID string

#### `mngs.gen.gen_timestamp()`
Generate a timestamp string.

**Returns:**
- Timestamp in format "YYYY-MM-DD-HH-MM-SS"

---

## io Module

### Core Functions

#### `mngs.io.save(obj, path, **kwargs)`
Save any Python object to file. Automatically detects format from extension.

**Supported formats:**
- `.npy`, `.npz`: NumPy arrays
- `.csv`: Pandas DataFrames
- `.pkl`, `.pickle`: Python objects (pickle)
- `.json`: JSON-serializable objects
- `.yaml`, `.yml`: YAML format
- `.txt`: Text files
- `.png`, `.jpg`, `.pdf`: Matplotlib figures
- `.h5`, `.hdf5`: HDF5 format
- `.xlsx`: Excel files
- `.joblib`: Joblib format
- `.pth`: PyTorch models

**Parameters:**
- `obj`: Object to save
- `path`: File path (directories created automatically)
- `**kwargs`: Format-specific options

**Example:**
```python
# Save various formats
mngs.io.save(numpy_array, "data.npy")
mngs.io.save(dataframe, "results.csv", index=False)
mngs.io.save(config_dict, "config.json", indent=2)
mngs.io.save(figure, "plot.png", dpi=300)
```

#### `mngs.io.load(path, **kwargs)`
Load any file based on extension.

**Parameters:**
- `path`: File path or glob pattern
- `**kwargs`: Format-specific options

**Returns:**
- Loaded object

**Example:**
```python
# Load single file
data = mngs.io.load("data.npy")

# Load with glob pattern
all_csvs = mngs.io.load("results/*.csv")
```

### Specialized Functions

#### `mngs.io.glob(pattern, recursive=False)`
Find files matching a pattern.

**Parameters:**
- `pattern`: Glob pattern
- `recursive`: Search recursively (default: False)

**Returns:**
- List of matching file paths

#### `mngs.io.load_configs(config_dir="./config")`
Load all YAML configuration files from a directory.

**Parameters:**
- `config_dir`: Directory containing config files

**Returns:**
- DotDict with merged configurations

---

## plt Module

### Enhanced Plotting

#### `mngs.plt.subplots(*args, **kwargs)`
Create matplotlib subplots with automatic data tracking.

**Parameters:**
- Same as matplotlib.pyplot.subplots()

**Returns:**
- `fig`: Figure object
- `axes`: Axes object(s) with data tracking

**Special Features:**
- Automatically saves plotted data as CSV when saving figure
- Tracks all plot commands for reproducibility

**Example:**
```python
fig, ax = mngs.plt.subplots(figsize=(8, 6))
ax.plot(x, y, label="data")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
mngs.io.save(fig, "plot.png")  # Also saves plot_data.csv
```

### Color Utilities

#### `mngs.plt.color.get_colors(n, cmap="viridis")`
Get n evenly spaced colors from a colormap.

**Parameters:**
- `n`: Number of colors
- `cmap`: Matplotlib colormap name

**Returns:**
- List of color tuples

---

## pd Module

### DataFrame Creation

#### `mngs.pd.force_df(obj)`
Force any object into a pandas DataFrame.

**Parameters:**
- `obj`: Object to convert (list, dict, Series, array, etc.)

**Returns:**
- pandas DataFrame

### Column Operations

#### `mngs.pd.merge_columns(df, cols, sep="_", new_col=None)`
Merge multiple columns into one.

**Parameters:**
- `df`: DataFrame
- `cols`: List of column names
- `sep`: Separator (default: "_")
- `new_col`: New column name

**Returns:**
- DataFrame with merged column

#### `mngs.pd.melt_cols(df, cols)`
Melt specific columns while keeping others as identifiers.

**Parameters:**
- `df`: DataFrame
- `cols`: Columns to melt

**Returns:**
- Melted DataFrame

### Data Manipulation

#### `mngs.pd.find_indi(df, **kwargs)`
Find indices matching conditions.

**Parameters:**
- `df`: DataFrame
- `**kwargs`: Column-value pairs for filtering

**Returns:**
- Boolean index array

**Example:**
```python
indices = mngs.pd.find_indi(df, age=25, status="active")
filtered = df[indices]
```

### Coordinate Transformations

#### `mngs.pd.to_xyz(df, x_col, y_col, z_col)`
Convert DataFrame to xyz format.

#### `mngs.pd.from_xyz(xyz_df, z_col="value")`
Convert xyz format back to DataFrame.

---

## dsp Module

### Signal Generation

#### `mngs.dsp.demo_sig(sig_type="uniform", t_sec=1, fs=1000, freqs_hz=None, n_chs=19, batch_size=8)`
Generate demo signals for testing.

**Parameters:**
- `sig_type`: "uniform", "gauss", "periodic", "chirp", "ripple"
- `t_sec`: Duration in seconds
- `fs`: Sampling frequency
- `freqs_hz`: Frequencies for periodic signals
- `n_chs`: Number of channels
- `batch_size`: Batch size

**Returns:**
- `signal`: 3D array (batch, channels, time)
- `time`: Time array
- `fs`: Sampling frequency

### Filtering

#### `mngs.dsp.filt.bandpass(x, fs, bands, t=None)`
Apply bandpass filter.

**Parameters:**
- `x`: Signal (1D, 2D, or 3D array)
- `fs`: Sampling frequency
- `bands`: 2D array of [low, high] frequencies
- `t`: Time array (optional)

**Returns:**
- Filtered signal

#### `mngs.dsp.filt.lowpass(x, fs, cutoffs_hz, t=None)`
Apply lowpass filter.

#### `mngs.dsp.filt.highpass(x, fs, cutoffs_hz, t=None)`
Apply highpass filter.

### Spectral Analysis

#### `mngs.dsp.psd(x, fs, nperseg=None, method="welch")`
Compute power spectral density.

**Parameters:**
- `x`: Signal
- `fs`: Sampling frequency
- `nperseg`: Segment length
- `method`: "welch" or "periodogram"

**Returns:**
- `psd`: Power spectral density
- `freqs`: Frequency array

#### `mngs.dsp.hilbert(x)`
Compute Hilbert transform.

**Returns:**
- `phase`: Instantaneous phase
- `amplitude`: Instantaneous amplitude

### Time-Frequency Analysis

#### `mngs.dsp.wavelet(x, fs, freqs, method="morlet")`
Compute wavelet transform.

**Parameters:**
- `x`: Signal
- `fs`: Sampling frequency
- `freqs`: Frequencies to analyze
- `method`: Wavelet type

**Returns:**
- `cwt`: Complex wavelet transform
- `freqs`: Frequency array

### Phase-Amplitude Coupling

#### `mngs.dsp.pac(x, fs, pha_start_hz=2, pha_end_hz=20, amp_start_hz=60, amp_end_hz=160, **kwargs)`
Compute phase-amplitude coupling.

**Parameters:**
- `x`: Signal
- `fs`: Sampling frequency
- Phase and amplitude band parameters

**Returns:**
- `pac_values`: PAC matrix
- `phase_freqs`: Phase frequencies
- `amp_freqs`: Amplitude frequencies

### Signal Processing

#### `mngs.dsp.resample(x, src_fs, tgt_fs, t=None)`
Resample signal to new sampling rate.

#### `mngs.dsp.crop(x, window_length, overlap=0)`
Crop signal into windows.

#### `mngs.dsp.ensure_3d(x)`
Ensure signal is 3D (batch, channels, time).

### Normalization

#### `mngs.dsp.norm.z(x)`
Z-score normalization.

#### `mngs.dsp.norm.minmax(x, vmin=-1, vmax=1)`
Min-max normalization.

---

## stats Module

### Descriptive Statistics

#### `mngs.stats.describe(data, method="pandas")`
Comprehensive descriptive statistics.

**Returns:**
- Dictionary with mean, std, median, etc.

### Correlation Analysis

#### `mngs.stats.corr_test(x, y, method="pearson")`
Test correlation between variables.

**Parameters:**
- `x`, `y`: Variables to correlate
- `method`: "pearson", "spearman", or "kendall"

**Returns:**
- Dictionary with r, p-value, CI, etc.

#### `mngs.stats.calc_partial_corr(x, y, z)`
Calculate partial correlation controlling for z.

### Statistical Tests

#### `mngs.stats.brunner_munzel_test(x, y)`
Brunner-Munzel test for two samples.

#### `mngs.stats.smirnov_grubbs(data, alpha=0.05)`
Smirnov-Grubbs test for outliers.

### Multiple Comparisons

#### `mngs.stats.bonferroni_correction(p_values, alpha=0.05)`
Bonferroni correction for multiple comparisons.

#### `mngs.stats.fdr_correction(p_values, alpha=0.05, method="bh")`
False Discovery Rate correction.

### Formatting

#### `mngs.stats.p2stars(p, thresholds=None)`
Convert p-values to significance stars.

**Default thresholds:**
- p < 0.001: "***"
- p < 0.01: "**"
- p < 0.05: "*"
- p >= 0.05: "n.s."

---

## ai Module

### Classification

#### `mngs.ai.ClassificationReporter()`
Generate detailed classification reports.

**Methods:**
- `report(y_true, y_pred)`: Generate report

#### `mngs.ai.bACC(y_true, y_pred)`
Calculate balanced accuracy.

### Training Utilities

#### `mngs.ai.EarlyStopping(patience=10, min_delta=0.001)`
Early stopping for training loops.

**Methods:**
- `__call__(val_loss)`: Check if should stop
- `reset()`: Reset counter

### Generative AI

#### `mngs.ai.OpenAI(api_key, model="gpt-4", **kwargs)`
OpenAI API wrapper.

**Methods:**
- `generate(prompt, max_tokens=None)`: Generate text
- `calculate_cost(tokens)`: Calculate API cost

Similar classes available for:
- `mngs.ai.Anthropic`
- `mngs.ai.Google`
- `mngs.ai.Groq`
- `mngs.ai.DeepSeek`
- `mngs.ai.Perplexity`

### Scikit-learn Wrappers

#### `mngs.ai.sk`
Namespace for enhanced scikit-learn classifiers:
- `RandomForestClassifier`
- `XGBClassifier`
- `SVC`
- `LogisticRegression`
- etc.

---

## nn Module

### Signal Processing Layers (PyTorch)

#### `mngs.nn.BandPassFilter(bands, fs, seq_len)`
Trainable bandpass filter layer.

#### `mngs.nn.Hilbert()`
Hilbert transform layer.

#### `mngs.nn.PSD(fs, nperseg=None)`
Power spectral density layer.

#### `mngs.nn.PAC(fs, pha_bands, amp_bands)`
Phase-amplitude coupling layer.

#### `mngs.nn.Wavelet(fs, freqs)`
Wavelet transform layer.

---

## str Module

### String Utilities

#### `mngs.str.printc(text, c="red", bold=False)`
Print colored text.

**Colors**: red, green, yellow, blue, magenta, cyan, white

#### `mngs.str.squeeze_space(text)`
Replace multiple spaces with single space.

#### `mngs.str.gen_ID(n=8)`
Generate random ID string.

#### `mngs.str.gen_timestamp()`
Generate timestamp string.

---

## path Module

### Path Utilities

#### `mngs.path.find(pattern, recursive=True)`
Find files matching pattern.

#### `mngs.path.split(path)`
Split path into components.

**Returns:**
- Dictionary with dirname, basename, stem, ext

#### `mngs.path.increment_version(path)`
Increment version number in filename.

**Example:**
```python
mngs.path.increment_version("file_v001.txt")  # Returns: "file_v002.txt"
```

---

## decorators Module

### Function Decorators

#### `@mngs.decorators.cache_to_disk(cache_dir="./cache")`
Cache function results to disk.

#### `@mngs.decorators.numpy_fn`
Ensure function inputs/outputs are numpy arrays.

#### `@mngs.decorators.torch_fn`
Ensure function inputs/outputs are torch tensors.

#### `@mngs.decorators.batch_fn`
Apply function to batched inputs.

#### `@mngs.decorators.tqdm_wrap`
Add progress bar to function.

---

## Other Utilities

### dict Module

#### `mngs.dict.DotDict(dict)`
Dictionary with dot notation access.

```python
config = mngs.dict.DotDict({"model": {"lr": 0.001}})
print(config.model.lr)  # 0.001
```

### resource Module

#### `mngs.resource.get_gpu_usage()`
Get current GPU memory usage.

#### `mngs.resource.limit_RAM(gb)`
Limit RAM usage for current process.

### web Module

#### `mngs.web.search_pubmed(query, max_results=10)`
Search PubMed database.

#### `mngs.web.summarize_url(url)`
Get summary of web page content.

---

## Best Practices

### 1. Always use mngs.gen.start/close
```python
CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt)
try:
    # Your code here
    pass
finally:
    mngs.gen.close(CONFIG)
```

### 2. Let mngs handle paths
```python
# Don't do:
os.makedirs("output/plots", exist_ok=True)
plt.savefig("output/plots/fig.png")

# Do:
mngs.io.save(fig, "fig.png")  # Auto-creates directories
```

### 3. Use consistent data shapes
- Signals: (batch, channels, time)
- Images: (batch, channels, height, width)
- DataFrames: Rows are samples, columns are features

### 4. Leverage integration
```python
# Load data -> process -> analyze -> visualize -> save
data = mngs.io.load("signal.npy")
filtered = mngs.dsp.filt.bandpass(data, fs=1000, bands=[[1, 50]])
psd, freqs = mngs.dsp.psd(filtered, fs=1000)

fig, ax = mngs.plt.subplots()
ax.plot(freqs, psd[0, 0])
mngs.io.save(fig, "psd.png")  # Saves both image and data
```

---

## Version History

- v1.0.0 (2025-05-31): 100% test coverage achieved
- v0.9.0: Major refactoring and API stabilization
- v0.8.0: Added comprehensive examples
- v0.7.0: Enhanced documentation

---

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## License

MNGS is released under the MIT License.