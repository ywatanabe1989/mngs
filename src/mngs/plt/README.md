<!-- ---
!-- Timestamp: 2025-01-15 10:42:04
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/README.md
!-- --- -->


# `mngs.plt` Quick Start Guide

The `mngs.plt` module is part of the `mngs` package and provides a suite of plotting utilities built on top of Matplotlib and Seaborn. It offers enhanced plotting functions, axis manipulation utilities, data tracking, and helper functions to create publication-quality figures with minimal effort.

You may also want to import Matplotlib and Seaborn for additional plotting capabilities:

```python
import matplotlib.pyplot as plt
import seaborn as sns
```

## Overview

The `mngs.plt` module provides:

- **Enhanced Figure and Axis Wrappers**: Easily create figures and axes that keep track of plotting history, allowing for data extraction and manipulation.
- **Axis Utilities**: Functions to adjust axis properties, such as labels, ticks, spines, and aspect ratios.
- **Plotting Functions**: Extended plotting functions that support additional features like confidence intervals, raster plots, and more.
- **Seaborn Integration**: Seamless integration with Seaborn plotting functions with added functionalities.
- **Color Management**: Tools to manage colors and colormaps for consistent and attractive visualizations.
- **Configuration Functions**: Setup functions to configure Matplotlib settings globally.
- **Data Tracking for SigmaPlot Compatibility**: Ability to extract plotting data in a format compatible with SigmaPlot.

## Examples
```python
# Import the mngs package
import mngs
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Configure Matplotlib settings using mngs.plt's configure_mpl
plt, CC = mngs.plt.configure_mpl(
    plt,
    fig_size_mm=(160, 100),  # Figure size in millimeters
    dpi_display=100,         # Display DPI
    dpi_save=300,            # Save DPI when exporting figures
    fontsize='medium',       # Base font size
    hide_top_right_spines=True,
    line_width=1.0,
    verbose=False
)

# Create a figure and axis using mngs.plt's enhanced subplots function
fig, ax = mngs.plt.subplots()

# Plot a simple sine wave
x = np.linspace(0, 10, 100)
y = np.sin(x)
ax.plot(x, y, label='Sine Wave')  # Plot the sine wave

# Plot cosine points
ax.scatter(x, np.cos(x), color='red', label='Cosine Points')  # Scatter plot of cosine values

# Customize the plot
ax.set_xyt(x='Time (s)', y='Amplitude', t='Trigonometric Functions')  # Set labels and title
ax.legend()  # Add a legend
ax.set_ticks(
    xvals=[0, 2, 4, 6, 8, 10],            # Set custom x-tick positions
    xticks=['0s', '2s', '4s', '6s', '8s', '10s']  # Set custom x-tick labels
)
ax.rotate_labels(x=45)  # Rotate x-axis labels for better readability
ax.hide_spines(top=True, right=True)  # Hide top and right spines

# Use mngs.plt's plot_ function to plot data with confidence intervals
# Generate sample data
data = np.random.randn(100, 50)  # 100 samples of 50 observations each

# Plot the mean with standard deviation
ax.plot_(data, line='mean', fill='std', n=1, label='Mean ± 1 SD')

# Plot the median with interquartile range
ax.plot_(data, line='median', fill='iqr', label='Median ± IQR')

# Add a legend and update the title
ax.legend()
ax.set_xyt(t='Statistical Measures with Confidence Intervals')

# Use Seaborn integration with mngs.plt
df = sns.load_dataset('tips')  # Load a sample dataset

# Create a violin plot
ax = mngs.plt.subplots()[1]  # Create a new axis for the violin plot
ax.sns_violinplot(data=df, x='day', y='total_bill', hue='sex', split=True)

# Customize the violin plot
ax.set_xyt(x='Day of Week', y='Total Bill', t='Total Bill Distribution by Day and Gender')

# Extract plotting data for SigmaPlot compatibility
plot_data = ax.to_sigma()
print(plot_data)  # Display the extracted data

# Save the extracted data to a CSV file for use with SigmaPlot
plot_data.to_csv('plot_data_for_sigmaplot.csv', index=False)
print("Plot data saved for SigmaPlot.")

# Save figure with sigmaplot-compatible csv format
mngs.io.save(fig, "plot_data_for_sigmplot.jpg")

# Create a raster plot (commonly used in neuroscience for spike trains)
# Simulate spike times for 3 neurons
spike_times = [
    np.random.uniform(0, 10, size=50),  # Neuron 1 spike times
    np.random.uniform(0, 10, size=30),  # Neuron 2 spike times
    np.random.uniform(0, 10, size=40)   # Neuron 3 spike times
]

# Create a new figure and axis for the raster plot
fig_raster, ax_raster = mngs.plt.subplots()

# Plot the raster
ax_raster.raster(spike_times)

# Customize the raster plot
ax_raster.set_xyt(x='Time (s)', y='Neuron Index', t='Raster Plot of Neuron Spiking Activity')

# Plot a confusion matrix using mngs.plt's conf_mat function
# Sample confusion matrix data
confusion_matrix = np.array([[35, 5], [3, 57]])
class_names = ['Negative', 'Positive']

# Create a new figure and axis for the confusion matrix
fig_conf, ax_conf = mngs.plt.subplots()

# Plot the confusion matrix
ax_conf.conf_mat(
    confusion_matrix,
    x_labels=class_names,
    y_labels=class_names,
    title='Confusion Matrix'
)

# Customize the confusion matrix plot
ax_conf.set_xyt(x='Predicted Label', y='True Label')

# Adjust the size of the axis using extend
ax_conf.extend(x_ratio=1.2)  # Extend the axis horizontally by 20%

# Shift the axis position using shift
ax_conf.shift(dx=1.0, dy=0.5)  # Shift the axis 1 cm to the right and 0.5 cm up

# Demonstrate sharing axes limits across subplots
# Create multiple axes
fig_multi, axes = mngs.plt.subplots(nrows=2)

# Generate data for both subplots
x_shared = np.linspace(0, 10, 100)
y1 = np.sin(x_shared)
y2 = np.cos(x_shared)

# Plot data on both axes
axes[0].plot(x_shared, y1, label='Sine')
axes[1].plot(x_shared, y2, label='Cosine')

# Share x and y limits
axes.sharex()
axes.sharey()

# Customize each subplot
axes[0].set_xyt(y='Amplitude', t='Sine Wave')
axes[1].set_xyt(x='Time (s)', y='Amplitude', t='Cosine Wave')

# Add legends
axes[0].legend()
axes[1].legend()

# Show the plots
plt.show()
```

## Contact
Yusuke Watanabe (ywatanabe@alumni.u-tokyo.ac.jp)
