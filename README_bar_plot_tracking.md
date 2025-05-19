# Bar Plot Data Tracking in MNGS

## Issue with CSV Export

We identified and fixed issues with bar plot data tracking in the mngs plotting library:

1. Bar plots weren't properly tracking data for CSV export
2. This caused empty/minimal CSV files (1 byte) when saving plots
3. The AxisWrapper wrapper was using incompatible array conversion

## Fixes Implemented

### 1. Data Tracking for Bar Plots

We've enhanced the `bar()` method in `MatplotlibPlotMixin` to:
- Track all bar data properties (x, height, width, bottom)
- Calculate bar positions for better visualization
- Include style attributes in tracked data
- Store everything in a structured DataFrame

### 2. CSV Export Formatter

Updated the `_format_bar.py` formatter to:
- Handle both the new and legacy tracking formats
- Properly process array and scalar data
- Calculate derived values (like bar edges)
- Include all relevant parameters for analysis

### 3. Improved Error Handling

Modified the `export_as_csv.py` function to:
- Always return a DataFrame (empty if needed) instead of None
- Catch and report formatting errors without failing
- Provide diagnostic information when combination fails
- Support partial exports when some plots fail

## Usage in gPAC Visualization Script

To fix issues with the gPAC visualization script:

```bash
# Run the provided fix script directly:
python fix_visualization_script_with_data_tracking.py /path/to/gPAC-paper-with-code/scripts/1_data_preparation/visualize_parameter_space.py
```

The script makes these changes:
1. Replaces `np.array(axes).flatten()` with `list(axes.flatten())`
2. Replaces `axes = axes.flatten()` with `list(axes.flatten())`
3. Adds tracking IDs to bar plots: `id="bar_i"` 

## Best Practices for Bar Plots

For best results with bar plots and data tracking:

1. Always include an ID when creating bar plots:
   ```python
   ax.bar(x, height, id="my_data")
   ```

2. If creating many bar plots in a loop, use the loop variable:
   ```python
   for i, param in enumerate(params):
       ax = axes[i]
       ax.bar([1, 2, 3], values, id=f"param_{param}")
   ```

3. Check your exported CSV data includes the expected columns:
   ```python
   df = fig.export_as_csv()
   print(df.columns)  # Should show bar_data columns
   print(df.shape)    # Check for non-empty data
   ```

These changes ensure proper data tracking for all bar plots, fixing the issues with empty CSV exports and awkward visualizations.