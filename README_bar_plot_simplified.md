# Simplified Bar Plot Data Tracking

## Overview

The bar plot data tracking system has been updated to focus only on essential data (x and y values), making CSV exports more concise and easier to use.

## Changes Made

1. **Simplified `bar()` Method**:
   - Now tracks only x and height (y) values
   - Removed extra metadata like width, bottom, positions
   - Maintains full plotting functionality

2. **Streamlined CSV Formatter**:
   - The `_format_bar.py` formatter now exports only x and y data
   - Consistent naming across all bar plots (x, y)
   - Smaller, more manageable CSV files

3. **Enhanced ID Generation**:
   - Auto-generates meaningful IDs when none provided
   - Format: `method_counter` (e.g., `bar_0`, `bar_1`)
   - IDs are preserved in CSV column names

## Benefits

- **Cleaner CSV Files**: Only the essential data is exported
- **Smaller File Sizes**: Reduced by excluding non-essential metadata
- **Consistent Naming**: All bar plots use the same x/y column pattern
- **Backwards Compatible**: Works with existing code
- **Better for Analysis**: Easier to work with the exported data

## Example

For a simple bar plot:
```python
ax.bar([1, 2, 3], [4, 5, 6])
```

### Before:
CSV with many columns:
```
bar_0_x, bar_0_height, bar_0_bottom, bar_0_width, bar_0_x_left, bar_0_x_right, bar_0_y_bottom, bar_0_y_top
```

### After:
Simplified CSV:
```
bar_0_x, bar_0_y
```

## Usage in Visualization Scripts

No changes needed to existing code - the simplification happens automatically when CSV files are exported.

For new scripts, you can use either approach:
```python
# Auto-generated ID (bar_0, bar_1, etc.)
ax.bar([1, 2, 3], [4, 5, 6])

# Or explicit ID
ax.bar([1, 2, 3], [4, 5, 6], id="my_data")
```

Both will now export just the x and y values to the CSV, making the data more concise and accessible.