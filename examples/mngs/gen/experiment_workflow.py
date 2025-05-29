#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-30 02:05:00 (Claude)"
# File: examples/mngs/gen/experiment_workflow.py

"""
Complete experiment workflow with mngs.gen

This example demonstrates:
- Setting up reproducible experiments with mngs.gen.start
- Automatic logging and output management
- Random seed management
- Proper cleanup with mngs.gen.close
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mngs

# Parameters
PARAMS = {
    "seed": 42,
    "n_samples": 1000,
    "n_features": 20,
    "noise_level": 0.1,
}

def generate_synthetic_data(n_samples, n_features, noise_level):
    """Generate synthetic dataset for demonstration."""
    X = np.random.randn(n_samples, n_features)
    # Create target with linear relationship plus noise
    true_weights = np.random.randn(n_features)
    y = X @ true_weights + noise_level * np.random.randn(n_samples)
    return X, y, true_weights

def analyze_data(X, y):
    """Perform basic data analysis."""
    # Calculate correlations
    correlations = []
    for i in range(X.shape[1]):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        correlations.append(corr)
    
    # Create summary statistics
    summary = {
        'n_samples': len(y),
        'n_features': X.shape[1],
        'y_mean': float(np.mean(y)),
        'y_std': float(np.std(y)),
        'max_correlation': float(np.max(np.abs(correlations))),
        'mean_correlation': float(np.mean(np.abs(correlations)))
    }
    
    return correlations, summary

def create_visualizations(X, y, correlations, plt):
    """Create analysis visualizations."""
    # Figure 1: Data overview
    fig1, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Target distribution
    axes[0, 0].hist(y, bins=30, alpha=0.7, color='blue')
    axes[0, 0].set_xlabel('Target Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Target Distribution')
    
    # Feature correlations
    axes[0, 1].bar(range(len(correlations)), correlations, alpha=0.7)
    axes[0, 1].set_xlabel('Feature Index')
    axes[0, 1].set_ylabel('Correlation with Target')
    axes[0, 1].set_title('Feature Correlations')
    
    # Sample of data
    axes[1, 0].scatter(X[:, 0], y, alpha=0.5, s=10)
    axes[1, 0].set_xlabel('Feature 0')
    axes[1, 0].set_ylabel('Target')
    axes[1, 0].set_title('Feature 0 vs Target')
    
    # Feature means
    feature_means = np.mean(X, axis=0)
    axes[1, 1].plot(feature_means, 'o-', alpha=0.7)
    axes[1, 1].set_xlabel('Feature Index')
    axes[1, 1].set_ylabel('Mean Value')
    axes[1, 1].set_title('Feature Mean Values')
    
    plt.tight_layout()
    mngs.io.save(fig1, "data_analysis_overview.png")
    
    # Figure 2: Correlation heatmap
    fig2, ax = plt.subplots(figsize=(8, 6))
    # Simple correlation matrix visualization
    corr_matrix = np.corrcoef(X.T)
    im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax.set_title('Feature Correlation Matrix')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Feature Index')
    plt.colorbar(im, ax=ax, label='Correlation')
    mngs.io.save(fig2, "correlation_matrix.png")
    
    return fig1, fig2

def main(CONFIG, plt):
    """Main experiment function."""
    print("🚀 Starting Experiment")
    print(f"📊 Experiment ID: {CONFIG.ID}")
    print(f"📁 Output Directory: {CONFIG.SDIR}")
    
    # 1. Generate data
    print("\n1. Generating synthetic data...")
    X, y, true_weights = generate_synthetic_data(
        PARAMS["n_samples"],
        PARAMS["n_features"],
        PARAMS["noise_level"]
    )
    print(f"   Generated data shape: X={X.shape}, y={y.shape}")
    
    # 2. Save raw data
    print("\n2. Saving raw data...")
    mngs.io.save({
        'X': X,
        'y': y,
        'true_weights': true_weights
    }, "raw_data.npz")
    
    # Also save as CSV for easy inspection
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y
    mngs.io.save(df, "dataset.csv", index=False)
    
    # 3. Analyze data
    print("\n3. Analyzing data...")
    correlations, summary = analyze_data(X, y)
    mngs.io.save(summary, "analysis_summary.json")
    print(f"   Max correlation: {summary['max_correlation']:.3f}")
    
    # 4. Create visualizations
    print("\n4. Creating visualizations...")
    fig1, fig2 = create_visualizations(X, y, correlations, plt)
    print("   Saved 2 figures")
    
    # 5. Generate report
    print("\n5. Generating report...")
    report = f"""Experiment Report
=================

Experiment ID: {CONFIG.ID}
Date: {CONFIG.START_TIME}

Parameters:
-----------
{mngs.str.print_block(PARAMS, return_str=True)}

Results Summary:
---------------
- Number of samples: {summary['n_samples']}
- Number of features: {summary['n_features']}
- Target mean: {summary['y_mean']:.3f}
- Target std: {summary['y_std']:.3f}
- Max feature correlation: {summary['max_correlation']:.3f}
- Mean absolute correlation: {summary['mean_correlation']:.3f}

Files Generated:
---------------
- raw_data.npz: NumPy arrays of X, y, and true weights
- dataset.csv: Combined dataset in CSV format
- analysis_summary.json: Statistical summary
- data_analysis_overview.png: 4-panel visualization
- correlation_matrix.png: Feature correlation heatmap

Conclusion:
----------
Successfully generated and analyzed synthetic dataset with linear relationships.
All outputs saved to: {CONFIG.SDIR}
"""
    
    mngs.io.save(report, "experiment_report.md")
    print("\n✅ Experiment completed successfully!")
    
    return summary

if __name__ == "__main__":
    # Initialize environment with mngs
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys=sys,
        plt=plt,
        seed=PARAMS["seed"],
        sdir_suffix="synthetic_data_analysis"
    )
    
    try:
        # Run main experiment
        results = main(CONFIG, plt)
        
        # Additional: demonstrate CONFIG usage
        print(f"\n📋 CONFIG contains:")
        print(f"   - ID: {CONFIG.ID}")
        print(f"   - PID: {CONFIG.PID}")
        print(f"   - START_TIME: {CONFIG.START_TIME}")
        print(f"   - SDIR: {CONFIG.SDIR}")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        raise
    
    finally:
        # Always close to ensure proper cleanup
        mngs.gen.close(CONFIG)
        print("\n🔚 Environment closed properly")