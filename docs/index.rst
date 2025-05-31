.. MNGS documentation master file

MNGS - Python Utility Package for Scientific Computing
======================================================

.. image:: https://img.shields.io/pypi/v/mngs.svg
   :target: https://pypi.org/project/mngs/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/mngs.svg
   :target: https://pypi.org/project/mngs/
   :alt: Python versions

**MNGS** (pronounced "monogusa", meaning "lazy" in Japanese) is a comprehensive Python utility package designed to streamline scientific computing workflows. It provides standardized tools for I/O operations, signal processing, plotting, statistics, and more.

Key Features
------------

* **Comprehensive I/O**: Support for 20+ file formats with unified interface
* **Signal Processing**: Advanced DSP tools for filtering, spectral analysis, and more
* **Enhanced Plotting**: matplotlib wrapper with automatic data tracking and export
* **Statistical Analysis**: Common statistical tests and utilities
* **Reproducibility**: Built-in experiment tracking and seed management
* **GPU Support**: Seamless integration with PyTorch for GPU computing

Installation
------------

.. code-block:: bash

   pip install mngs

Quick Start
-----------

.. code-block:: python

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
   data = mngs.io.load("data.pkl")
   results = process_data(data)
   mngs.io.save(results, "results.pkl")

   # Clean up and finalize
   mngs.gen.close(CONFIG)

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   installation
   mngs_guidelines/MNGS_COMPLETE_REFERENCE
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules
   api/mngs.gen
   api/mngs.io
   api/mngs.plt
   api/mngs.dsp
   api/mngs.stats
   api/mngs.ai

.. toctree::
   :maxdepth: 2
   :caption: Agent Documentation

   mngs_guidelines/README
   mngs_guidelines/agent_guidelines/00_why_use_mngs
   mngs_guidelines/agent_guidelines/01_quick_start
   mngs_guidelines/agent_guidelines/02_core_concepts
   mngs_guidelines/agent_guidelines/03_module_overview
   mngs_guidelines/agent_guidelines/04_common_workflows

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`