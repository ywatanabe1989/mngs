Getting Started
===============

This guide will help you get started with MNGS for your scientific computing projects.

Overview
--------

MNGS (monogusa) is designed to make scientific Python programming more efficient by providing:

1. **Standardized I/O operations** - One interface for all file formats
2. **Experiment management** - Automatic logging and configuration tracking
3. **Enhanced plotting** - matplotlib with superpowers
4. **Signal processing** - Comprehensive DSP toolkit
5. **Statistical utilities** - Common tests and analyses

Basic Usage Pattern
-------------------

The typical MNGS workflow follows this pattern:

.. code-block:: python

   #!/usr/bin/env python3
   import sys
   import matplotlib.pyplot as plt
   import mngs

   # Initialize experiment environment
   CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
       sys, plt,
       file=__file__,
       verbose=False
   )

   # Your scientific code here
   # ...

   # Clean up and save logs
   mngs.gen.close(CONFIG)

Key Concepts
------------

Experiment Management
~~~~~~~~~~~~~~~~~~~~~

Every MNGS script starts with ``mngs.gen.start()`` which:

- Generates a unique experiment ID
- Sets up logging to capture all output
- Configures matplotlib for publication-quality plots
- Fixes random seeds for reproducibility
- Creates an output directory structure

The output directory structure looks like::

   your_script_out/
   ├── RUNNING/
   │   └── {ID}/
   │       ├── logs/
   │       │   ├── stdout.log
   │       │   └── stderr.log
   │       └── CONFIGS/
   └── FINISHED_SUCCESS/  # After successful completion
       └── {ID}/
           └── ... (same structure)

Unified I/O Interface
~~~~~~~~~~~~~~~~~~~~~

MNGS provides a single interface for loading and saving data:

.. code-block:: python

   # Save any data type - format detected from extension
   mngs.io.save(data, "output.pkl")    # Pickle
   mngs.io.save(data, "output.npy")    # NumPy
   mngs.io.save(data, "output.csv")    # Pandas CSV
   mngs.io.save(data, "output.json")   # JSON
   mngs.io.save(data, "output.mat")    # MATLAB

   # Load with the same simple interface
   data = mngs.io.load("input.pkl")

Enhanced Plotting
~~~~~~~~~~~~~~~~~

MNGS wraps matplotlib with additional features:

.. code-block:: python

   fig, ax = mngs.plt.subplots()
   ax.plot([1, 2, 3], [1, 4, 9], label="data")
   
   # Automatic tracking of plotted data
   print(ax.data)  # Access the plotted data
   
   # Export plot data as CSV alongside the image
   fig.save("plot.png")  # Also creates plot_data.csv

Next Steps
----------

- See the :doc:`tutorials/index` for detailed examples
- Browse the :doc:`api/modules` for complete API documentation
- Check out the example scripts in the repository's ``examples/`` directory