#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-07 18:52:00 (ywatanabe)"
# File: ./examples/gen/utilities_demo.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/gen/utilities_demo.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Demonstrates mngs.gen utility functions
  - Shows logging and configuration management
  - Demonstrates reproducibility features
  - Saves various utility outputs

Dependencies:
  - scripts:
    - None
  - packages:
    - mngs
    - numpy
IO:
  - input-files:
    - None

  - output-files:
    - ./examples/gen/utilities_demo_out/logs/*
    - ./examples/gen/utilities_demo_out/config.yaml
    - ./examples/gen/utilities_demo_out/system_info.txt
"""

"""Imports"""
import numpy as np
import time
import mngs

"""Parameters"""
SEED = 42

"""Functions"""
def demonstrate_utilities():
    """Demonstrate various mngs.gen utilities."""
    # Start with all features
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        __FILE__, 
        sys_=True,  # Redirect stdout/stderr to log
        plt_=True,  # Enable plotting
        verbose=True
    )
    
    print("MNGS General Utilities Demonstration")
    print("=" * 50)
    
    # 1. Configuration management
    print("\n1. Configuration Management")
    print("-" * 30)
    print(f"Script name: {CONFIG.ID}")
    print(f"Output directory: {CONFIG.SDIR}")
    print(f"Log directory: {CONFIG.SDIR}/logs")
    print(f"Timestamp: {CONFIG.NOW}")
    
    # 2. Reproducibility
    print("\n2. Reproducibility Features")
    print("-" * 30)
    
    # Fix random seeds
    mngs.reproduce.fix_seeds(SEED)
    print(f"Fixed random seeds to: {SEED}")
    
    # Test reproducibility
    arr1 = np.random.randn(5)
    mngs.reproduce.fix_seeds(SEED)
    arr2 = np.random.randn(5)
    print(f"Arrays equal after seed reset: {np.array_equal(arr1, arr2)}")
    
    # Generate unique IDs
    unique_id = mngs.reproduce.gen_ID()
    timestamp = mngs.reproduce.gen_timestamp()
    print(f"Unique ID: {unique_id}")
    print(f"Timestamp: {timestamp}")
    
    # 3. Type checking and conversion
    print("\n3. Type Utilities")
    print("-" * 30)
    
    # Check various types
    test_list = [1, 2, 3]
    test_array = np.array([1, 2, 3])
    
    print(f"Is [1,2,3] a list? {mngs.gen.is_list(test_list)}")
    print(f"Is np.array a list? {mngs.gen.is_list(test_array)}")
    print(f"Is [1,2,3] array-like? {mngs.gen.is_array_like(test_list)}")
    
    # 4. System information
    print("\n4. System Information")
    print("-" * 30)
    
    # Get system specs
    specs = mngs.resource.get_specs()
    print("System specifications:")
    for key, value in specs.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    # Save system info
    system_info = "System Information\n" + "=" * 30 + "\n"
    for key, value in specs.items():
        system_info += f"{key}: {value}\n"
    mngs.io.save(system_info, "system_info.txt")
    
    # 5. Timing utilities
    print("\n5. Timing and Performance")
    print("-" * 30)
    
    # Simple timing
    start_time = time.time()
    
    # Simulate some work
    data = np.random.randn(1000, 1000)
    result = np.linalg.svd(data, compute_uv=False)
    
    elapsed = time.time() - start_time
    print(f"SVD computation took: {elapsed:.4f} seconds")
    
    # 6. Path utilities
    print("\n6. Path Management")
    print("-" * 30)
    
    # Clean paths
    messy_path = "//some//messy///path//"
    clean_path = mngs.path.clean(messy_path)
    print(f"Original: '{messy_path}'")
    print(f"Cleaned: '{clean_path}'")
    
    # Get module path
    module_path = mngs.path.get_module_path(mngs)
    print(f"MNGS module path: {module_path}")
    
    # 7. Color utilities for terminal output
    print("\n7. Colored Terminal Output")
    print("-" * 30)
    
    print(mngs.str.color_text("Success message", "green"))
    print(mngs.str.color_text("Warning message", "yellow"))
    print(mngs.str.color_text("Error message", "red"))
    print(mngs.str.color_text("Info message", "blue"))
    
    # Summary
    print("\n" + "=" * 50)
    print("Utilities demonstrated:")
    print("✓ Configuration management")
    print("✓ Reproducibility (seeds, IDs)")
    print("✓ Type checking")
    print("✓ System information")
    print("✓ Timing utilities")
    print("✓ Path management")
    print("✓ Colored output")
    
    # Close and save logs
    mngs.gen.close(CONFIG)
    print(f"\n✅ Demo completed! Check outputs in: {CONFIG.SDIR}")

"""Main"""
if __name__ == "__main__":
    demonstrate_utilities()

"""EOF"""