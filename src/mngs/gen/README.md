<!-- ---
!-- Timestamp: 2025-01-15 10:53:31
!-- Author: ywatanabe
!-- File: ./src/mngs/gen/README.md
!-- --- -->

# `mngs.gen` Quick Start Guide

The `mngs.gen` module is a collection of general-purpose utility functions and classes designed to simplify common programming tasks in data science and machine learning workflows. This guide will introduce you to some of the key functions and show you how to use them with examples.

# [`mngs.gen`](https://github.com/ywatanabe1989/mngs/tree/main/src/mngs/gen/)

## Quick Start
```python
# Import necessary modules
import mngs
import sys
import numpy as np
import matplotlib.pyplot as plt

# Initialize the environment using mngs.gen.start
# This function sets up logging, fixes random seeds, configures matplotlib, and returns CONFIG and other variables
CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
    sys,        # System module for I/O redirection
    plt,        # Matplotlib pyplot module for plotting configuration
    verbose=True  # Set to False to suppress detailed output
)

# Your main code goes here
# For example, generate some data and plot it
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

plt.plot(x, y, label='Sine Wave')
plt.title('Sine Wave Plot')
plt.xlabel('Angle [rad]')
plt.ylabel('Sin(x)')
plt.legend()

# Save the figure using mngs.io.save
mngs.io.save(plt, 'sine_wave_plot.png') 

# See mngs.plt.subplots to automatic data tracking and saving in a sigmaplot-compatible format

# Finalize the script using mngs.gen.close
# This function handles cleanup tasks, saves configurations, and can send notifications if enabled
mngs.gen.close(CONFIG)
```

This script demonstrates the basic usage of `mngs.gen.start` and `mngs.gen.close` for initializing and finalizing your environment when running scripts using the `mngs` package.

- **`mngs.gen.start`**:
  - Sets up logging to capture stdout and stderr.
  - Fixes random seeds for reproducibility.
  - Configures Matplotlib settings.
  - Returns a configuration dictionary (`CONFIG`) and other variables for use in your script.
  
- **`mngs.gen.close`**:
  - Handles cleanup tasks such as flushing output streams.
  - Saves configuration settings and logs.
  - Optionally sends notifications upon script completion.

By wrapping your main code between `mngs.gen.start` and `mngs.gen.close`, you ensure that your script has a consistent environment and that all resources are properly managed.

**Note**: Replace `'sine_wave_plot.png'` with your desired file path or name for saving the plot.


## Contact
Yusuke Watanabe (ywata1989@gmail.com)

For more information and updates, please visit the [mngs GitHub repository](https://github.com/ywatanabe1989/mngs).
