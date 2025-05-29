# MNGS Common Workflows

## 1. Scientific Experiment Workflow

### Complete ML Experiment
```python
#!/usr/bin/env python3
import sys
import matplotlib.pyplot as plt
import mngs
import torch
import numpy as np

# Load configuration
CONFIG = mngs.io.load_configs()

# Start experiment
CONFIG_mngs, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
    sys, plt, 
    seed=CONFIG.seed,
    sdir_suffix=f"{CONFIG.model.name}_{CONFIG.training.lr}"
)

# Setup
model = MyModel(CONFIG.model)
data = mngs.io.load(CONFIG.data.train_path)

# Training loop
history = {"loss": [], "accuracy": []}

for epoch in range(CONFIG.training.epochs):
    loss, acc = train_epoch(model, data)
    history["loss"].append(loss)
    history["accuracy"].append(acc)
    
    # Save checkpoint
    if epoch % 10 == 0:
        mngs.io.save({
            "model": model.state_dict(),
            "epoch": epoch,
            "history": history
        }, f"checkpoint_epoch_{epoch}.pth")
    
    # Plot progress
    if epoch % 5 == 0:
        fig, ax = mngs.plt.subplots()
        ax.plot(history["loss"], label="Loss")
        ax.set_xyt("Epoch", "Loss", "Training Progress")
        mngs.io.save(fig, f"training_progress_{epoch}.png")

# Save final results
mngs.io.save(model.state_dict(), "final_model.pth")
mngs.io.save(history, "training_history.json")

# Close
mngs.gen.close(CONFIG_mngs)
```

## 2. Data Analysis Workflow

### EDA and Statistical Analysis
```python
#!/usr/bin/env python3
import mngs
import pandas as pd
import numpy as np

# Load data
df = mngs.io.load("./data/experiment_results.csv")

# Basic preprocessing
df_clean = df.dropna()
df_rounded = mngs.pd.round(df_clean, 3)

# Statistical analysis
results = {}
for col in df_clean.select_dtypes(include=[np.number]).columns:
    stats = mngs.stats.describe(df_clean[col])
    results[col] = stats

# Correlation analysis
corr_results = []
for i, col1 in enumerate(df_clean.columns[:-1]):
    for col2 in df_clean.columns[i+1:]:
        result = mngs.stats.corr_test(df_clean[col1], df_clean[col2])
        corr_results.append({
            "var1": col1,
            "var2": col2,
            "r": result["r"],
            "p": result["p"],
            "stars": mngs.stats.p2stars(result["p"])
        })

corr_df = pd.DataFrame(corr_results)

# Visualization
fig, axes = mngs.plt.subplots(2, 2, figsize=(10, 8))

# Distribution plots
for ax, col in zip(axes.flat, df_clean.columns[:4]):
    ax.hist(df_clean[col], bins=30, alpha=0.7)
    ax.set_xyt(col, "Frequency", f"Distribution of {col}")

mngs.io.save(fig, "distributions.png")

# Save results
mngs.io.save(results, "statistical_summary.json")
mngs.io.save(corr_df, "correlation_analysis.csv")
```

## 3. Signal Processing Workflow

### EEG/Neural Signal Analysis
```python
#!/usr/bin/env python3
import mngs
import numpy as np

# Configuration
fs = 1000  # Sampling frequency
channels = ["Fz", "Cz", "Pz", "Oz"]

# Load signal data
raw_signal = mngs.io.load("./data/eeg_recording.npy")
# Shape: (n_channels, n_timepoints)

# Process each channel
processed_data = {}

for ch_idx, ch_name in enumerate(channels):
    signal = raw_signal[ch_idx, :]
    
    # Filtering
    theta = mngs.dsp.bandpass(signal, 4, 8, fs)
    alpha = mngs.dsp.bandpass(signal, 8, 13, fs)
    beta = mngs.dsp.bandpass(signal, 13, 30, fs)
    gamma = mngs.dsp.bandpass(signal, 30, 100, fs)
    
    # Phase-amplitude coupling
    pac_theta_gamma = mngs.dsp.pac(theta, gamma, fs)
    
    # Power spectral density
    freqs, psd = mngs.dsp.psd(signal, fs)
    
    # Store results
    processed_data[ch_name] = {
        "filtered": {
            "theta": theta,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma
        },
        "pac": pac_theta_gamma,
        "psd": {"frequencies": freqs, "power": psd}
    }

# Visualization
fig, axes = mngs.plt.subplots(len(channels), 2, figsize=(12, 8))

for idx, (ch_name, data) in enumerate(processed_data.items()):
    # Time series
    ax_time = axes[idx, 0]
    t = np.arange(len(signal)) / fs
    ax_time.plot(t[:1000], data["filtered"]["alpha"][:1000])
    ax_time.set_xyt("Time (s)", "Amplitude", f"{ch_name} - Alpha")
    
    # PSD
    ax_psd = axes[idx, 1]
    ax_psd.semilogy(data["psd"]["frequencies"], data["psd"]["power"])
    ax_psd.set_xyt("Frequency (Hz)", "Power", f"{ch_name} - PSD")

mngs.io.save(fig, "signal_analysis.png")
mngs.io.save(processed_data, "processed_signals.pkl")
```

## 4. Batch Processing Workflow

### Process Multiple Files
```python
#!/usr/bin/env python3
import mngs
from pathlib import Path
import pandas as pd

# Find all data files
data_files = list(Path("./data/subjects/").glob("sub-*/data.csv"))

# Process each file
all_results = []

for file_path in data_files:
    subject_id = file_path.parent.name
    
    # Load and process
    data = mngs.io.load(file_path)
    
    # Compute metrics
    result = {
        "subject": subject_id,
        "mean": data.mean(),
        "std": data.std(),
        "n_samples": len(data)
    }
    
    # Statistical test against baseline
    baseline = mngs.io.load("./data/baseline.csv")
    test_result = mngs.stats.corr_test(data, baseline)
    result.update({
        "correlation": test_result["r"],
        "p_value": test_result["p"],
        "significant": test_result["p"] < 0.05
    })
    
    all_results.append(result)
    
    # Save individual results
    mngs.io.save(result, f"./results/individual/{subject_id}_analysis.json")

# Aggregate results
summary_df = pd.DataFrame(all_results)
summary_stats = mngs.stats.describe(summary_df)

# Group analysis
fig, ax = mngs.plt.subplots()
ax.bar(summary_df["subject"], summary_df["correlation"])
ax.set_xyt("Subject", "Correlation", "Subject-wise Correlations")
ax.tick_params(axis='x', rotation=45)

# Save summary
mngs.io.save(summary_df, "all_subjects_summary.csv")
mngs.io.save(summary_stats, "group_statistics.json")
mngs.io.save(fig, "group_analysis.png")
```

## 5. Report Generation Workflow

### Automated Analysis Report
```python
#!/usr/bin/env python3
import mngs
import pandas as pd
from datetime import datetime

# Start with report tracking
CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
    sys, plt,
    sdir_suffix="analysis_report"
)

# Load and analyze data
data = mngs.io.load("./data/experiment_data.csv")
results = analyze_data(data)  # Your analysis function

# Generate figures
figures = []

# Figure 1: Overview
fig1, ax = mngs.plt.subplots()
ax.plot(results["timeline"], results["values"])
ax.set_xyt("Time", "Measurement", "Experiment Overview")
mngs.io.save(fig1, "figure_1_overview.png")
figures.append("figure_1_overview.png")

# Figure 2: Statistical Summary
fig2, axes = mngs.plt.subplots(2, 2)
# ... create multiple plots ...
mngs.io.save(fig2, "figure_2_statistics.png")
figures.append("figure_2_statistics.png")

# Generate report
report = {
    "metadata": {
        "date": datetime.now().isoformat(),
        "analyst": CONFIG.ID,
        "data_source": "./data/experiment_data.csv"
    },
    "summary": {
        "n_samples": len(data),
        "key_findings": results["findings"],
        "statistics": results["stats"]
    },
    "figures": figures,
    "conclusions": results["conclusions"]
}

# Save report components
mngs.io.save(report, "analysis_report.json")
mngs.io.save(results, "detailed_results.pkl")

# Create markdown summary
summary_md = f"""
# Analysis Report
**Date**: {report['metadata']['date']}
**ID**: {report['metadata']['analyst']}

## Key Findings
{report['summary']['key_findings']}

## Figures
See attached figures in output directory.

## Conclusions
{report['conclusions']}
"""

mngs.io.save(summary_md, "report_summary.md")

# Close with summary
mngs.gen.close(CONFIG)
```

## 6. Real-time Monitoring Workflow

### Live Data Processing
```python
#!/usr/bin/env python3
import mngs
import time
import numpy as np
from collections import deque

# Setup
buffer_size = 1000
data_buffer = deque(maxlen=buffer_size)

# Initialize plot
fig, ax = mngs.plt.subplots()
line, = ax.plot([], [])
ax.set_xlim(0, buffer_size)
ax.set_ylim(-1, 1)
ax.set_xyt("Sample", "Value", "Real-time Monitor")

# Monitoring loop
try:
    for i in range(10000):
        # Simulate data acquisition
        new_data = np.sin(i * 0.1) + np.random.normal(0, 0.1)
        data_buffer.append(new_data)
        
        # Update plot every 100 samples
        if i % 100 == 0:
            line.set_data(range(len(data_buffer)), list(data_buffer))
            plt.pause(0.01)
            
        # Save snapshot every 1000 samples
        if i % 1000 == 0:
            mngs.io.save(list(data_buffer), f"buffer_snapshot_{i}.npy")
            mngs.io.save(fig, f"monitor_snapshot_{i}.png")
            
except KeyboardInterrupt:
    print("Monitoring stopped by user")
    
# Save final state
mngs.io.save(list(data_buffer), "final_buffer.npy")
mngs.io.save(fig, "final_monitor.png")
```

## Tips for Workflows

### 1. Error Handling
```python
try:
    data = mngs.io.load("data.csv")
except FileNotFoundError:
    mngs.str.printc("Data file not found!", c="red")
    # Handle missing data gracefully
```

### 2. Progress Tracking
```python
from tqdm import tqdm

files = list(Path("./data").glob("*.csv"))
results = []

for file in tqdm(files, desc="Processing files"):
    data = mngs.io.load(file)
    result = process(data)
    results.append(result)
```

### 3. Memory Management
```python
# Process large datasets in chunks
chunk_size = 1000
for i in range(0, len(large_data), chunk_size):
    chunk = large_data[i:i+chunk_size]
    process_chunk(chunk)
    # Clear memory
    del chunk
```

### 4. Parallel Processing
```python
from multiprocessing import Pool
import mngs

def process_file(filepath):
    data = mngs.io.load(filepath)
    result = analyze(data)
    mngs.io.save(result, f"results/{filepath.stem}_result.json")
    return result

# Process files in parallel
with Pool() as pool:
    files = list(Path("./data").glob("*.csv"))
    results = pool.map(process_file, files)
```

## Next Steps

- Explore module-specific guides in `docs/modules/`
- Check out example scripts in `examples/`
- Read about advanced features in module documentation