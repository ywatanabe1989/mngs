#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-29 19:37:24 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/custom/test_matplotlib_compatibility.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/custom/test_matplotlib_compatibility.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import pytest


def compare_modules_recursively(
    mpl_module, mngs_module, max_depth=3, current_depth=0, path=""
):
    """Recursively compare matplotlib and mngs modules and their attributes."""
    if current_depth > max_depth:
        return {}

    results = {}

    # Get all attributes from both modules
    mpl_attrs = set(dir(mpl_module))
    mngs_attrs = set(dir(mngs_module))

    # Basic comparison at current level
    results[path] = {
        "missing_in_mngs": sorted(list(mpl_attrs - mngs_attrs)),
        "unique_to_mngs": sorted(list(mngs_attrs - mpl_attrs)),
        "common": sorted(list(mpl_attrs.intersection(mngs_attrs))),
    }

    # Recursively explore common attributes that might be modules/classes
    for attr_name in results[path]["common"]:
        if attr_name.startswith("_"):
            continue

        try:
            mpl_attr = getattr(mpl_module, attr_name)
            mngs_attr = getattr(mngs_module, attr_name)

            # Skip non-object types and callables
            if not isinstance(mpl_attr, object) or callable(mpl_attr):
                continue

            # Create new path for recursive call
            new_path = f"{path}.{attr_name}" if path else attr_name

            # Recursively compare this attribute
            sub_results = compare_modules_recursively(
                mpl_attr,
                mngs_attr,
                max_depth=max_depth,
                current_depth=current_depth + 1,
                path=new_path,
            )

            # Merge results
            results.update(sub_results)
        except (AttributeError, TypeError):
            continue

    return results


def generate_comprehensive_report():
    """Generate a comprehensive compatibility report between matplotlib.pyplot and mngs.plt."""
    import matplotlib.pyplot as plt
    import mngs.plt

    # Compare the modules
    report = compare_modules_recursively(plt, mngs.plt)

    # Compare figure and axes objects specifically
    mpl_fig, mpl_ax = plt.subplots()
    mngs_fig, mngs_ax = mngs.plt.subplots()

    report["figure_objects"] = compare_modules_recursively(
        mpl_fig, mngs_fig, path="figure"
    )
    report["axes_objects"] = compare_modules_recursively(
        mpl_ax, mngs_ax, path="axes"
    )

    return report


def export_report_to_markdown(report, filename="compatibility_report.md"):
    """Export the compatibility report to a markdown file."""
    with open(filename, "w") as file:
        file.write("# Matplotlib and mngs.plt Compatibility Report\n\n")

        for path, data in report.items():
            if not path:
                path = "root"

            file.write(f"## {path}\n\n")

            file.write("### Missing in mngs.plt\n")
            if data.get("missing_in_mngs", []):
                for attr in data["missing_in_mngs"]:
                    file.write(f"- {attr}\n")
            else:
                file.write("- None\n")

            file.write("\n### Unique to mngs.plt\n")
            if data.get("unique_to_mngs", []):
                for attr in data["unique_to_mngs"]:
                    file.write(f"- {attr}\n")
            else:
                file.write("- None\n")

            file.write("\n")


def test_fundamental_plotting():
    """Test basic plotting capabilities and ensure they work in mngs.plt."""
    import mngs.plt

    # Create test data
    xx = np.linspace(0, 10, 100)
    yy = np.sin(xx)

    # Basic line plot
    fig, ax = mngs.plt.subplots()
    line = ax.plot(xx, yy)
    assert line is not None

    # Scatter plot
    scatter = ax.scatter(xx[::10], yy[::10])
    assert scatter is not None

    # Bar plot
    bars = ax.bar(range(5), np.random.rand(5))
    assert bars is not None


def test_figure_and_axes_methods():
    """Test that common figure and axes methods are available."""
    import mngs.plt

    # Create figure and axes
    fig, ax = mngs.plt.subplots()

    # Test figure methods
    fig.suptitle("Test Figure")
    fig.tight_layout()

    # Test axes methods
    ax.set_title("Test Axes")
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.grid(True)

    # Test that these operations completed without error
    assert ax.get_title() == "Test Axes"
    assert ax.get_xlabel() == "X Label"
    assert ax.get_ylabel() == "Y Label"


def test_fallback_mechanism():
    """Test that mngs.plt can fall back to matplotlib for missing functionality."""
    import warnings

    import mngs.plt

    # Get the compatibility report
    report = generate_comprehensive_report()

    # Find some missing methods to test
    if "axes_objects" in report:
        missing_methods = report["axes_objects"].get("missing_in_mngs", [])

        if missing_methods:
            fig, ax = mngs.plt.subplots()

            # Try to use a method that might be missing but should work via fallback
            # For this test, we'll just check the first few missing methods
            for method_name in missing_methods[:3]:
                # Skip private methods and known problematic ones
                if method_name.startswith("_") or method_name in [
                    "callbacks",
                    "stale_callback",
                ]:
                    continue

                # Check if the original matplotlib axes has this method
                mpl_fig, mpl_ax = plt.subplots()
                if hasattr(mpl_ax, method_name) and callable(
                    getattr(mpl_ax, method_name)
                ):
                    # The method exists and is callable in matplotlib
                    # We should be able to call it through mngs.plt's fallback
                    with warnings.catch_warnings(record=True) as warning_list:
                        warnings.simplefilter("always")

                        # Attempt to access and call the method
                        # Just check it exists, we won't actually call it as we don't know the parameters
                        method = getattr(ax, method_name, None)
                        assert (
                            method is not None
                        ), f"Method {method_name} should be available via fallback"

                        # Check that a warning was issued
                        assert any(
                            "fallback" in str(w.message).lower()
                            for w in warning_list
                        ), f"No fallback warning issued for {method_name}"


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

    # Generate and export the report when run directly
    report = generate_comprehensive_report()
    export_report_to_markdown(
        report, os.path.join(__DIR__, "compatibility_report.md")
    )
    print(
        f"Report exported to {os.path.join(__DIR__, 'compatibility_report.md')}"
    )

# EOF