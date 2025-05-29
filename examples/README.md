# MNGS Examples

This directory contains practical examples demonstrating how to use the mngs framework effectively.

## Structure

The examples are organized to mirror the mngs module structure:

```
examples/
├── mngs/
│   ├── io/           # File I/O examples
│   ├── gen/          # Environment setup examples
│   └── plt/          # Enhanced plotting examples
└── README.md         # This file
```

## Running Examples

All examples can be run directly from the project root:

```bash
# Basic file operations
python examples/mngs/io/basic_file_operations.py

# Complete experiment workflow
python examples/mngs/gen/experiment_workflow.py

# Enhanced plotting capabilities
python examples/mngs/plt/enhanced_plotting.py
```

## Example Descriptions

### 1. `io/basic_file_operations.py`
Demonstrates fundamental file I/O operations:
- Loading and saving various file formats (numpy, pandas, json, yaml, text)
- Automatic directory creation
- Working with compressed data
- Handling collections and nested structures

### 2. `gen/experiment_workflow.py`
Shows a complete scientific experiment workflow:
- Setting up reproducible environment with mngs.gen.start
- Automatic logging and output management
- Random seed control for reproducibility
- Generating synthetic data and analysis
- Creating visualizations and reports
- Proper cleanup with mngs.gen.close

### 3. `plt/enhanced_plotting.py`
Illustrates advanced plotting features:
- Using mngs.plt.subplots for automatic data tracking
- Multi-panel figures with different plot types
- Statistical visualizations with error bars
- Custom styling and formatting
- Automatic CSV export of plotted data

## Key Features Demonstrated

1. **Automatic Output Management**: All examples save outputs to organized directories
2. **Data Tracking**: Plots automatically export their data as CSV files
3. **Reproducibility**: Examples show how to set random seeds and track experiments
4. **Error Handling**: Examples include proper try-finally blocks where appropriate
5. **Documentation**: Each example is thoroughly commented

## Output Structure

After running the examples, you'll find outputs organized as:

```
output/
├── arrays/           # NumPy arrays and compressed data
├── dataframes/       # CSV files from pandas DataFrames
├── configs/          # JSON and YAML configuration files
├── reports/          # Text and markdown reports
├── plots/            # PNG images AND their corresponding CSV data
└── collections/      # Combined datasets
```

## Tips for Using Examples

1. **Start Simple**: Begin with `basic_file_operations.py` to understand core concepts
2. **Check Outputs**: Always examine the generated files to understand what mngs creates
3. **Read Comments**: Examples include detailed comments explaining each step
4. **Modify and Experiment**: Feel free to modify examples for your needs
5. **Use as Templates**: These examples can serve as templates for your own scripts

## Integration with Your Projects

To use these patterns in your own work:

```python
# Minimal usage
import mngs
data = mngs.io.load("your_data.csv")
mngs.io.save(results, "output.json")

# Full workflow
import sys
import matplotlib.pyplot as plt
import mngs

CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt)
# Your experiment code here
mngs.gen.close(CONFIG)
```

## Next Steps

After exploring these examples:
1. Read the full documentation in `docs/mngs_guidelines/`
2. Check out the module-specific guides
3. Explore the test files for more usage patterns
4. Start building your own projects with mngs!

## Contributing

If you create useful examples, consider contributing them back to the project!