#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-29 19:37:24 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/custom/test_matplotlib_compatibility.py
# ----------------------------------------
import os

__FILE__ = "./tests/custom/test_matplotlib_compatibility.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

# Use non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')

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
            # Use hasattr to check if attribute can be accessed safely
            if not hasattr(mpl_module, attr_name) or not hasattr(mngs_module, attr_name):
                continue
                
            mpl_attr = getattr(mpl_module, attr_name, None)
            mngs_attr = getattr(mngs_module, attr_name, None)
            
            if mpl_attr is None or mngs_attr is None:
                continue

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
        except (AttributeError, TypeError, ValueError, Exception):
            # Skip any attributes that raise errors when accessed
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
    report["axes_objects"] = compare_modules_recursively(mpl_ax, mngs_ax, path="axes")

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


def test_subplots_creation():
    """Test various ways to create subplots."""
    import mngs.plt

    # Single subplot
    fig, ax = mngs.plt.subplots()
    assert fig is not None
    assert ax is not None

    # Multiple subplots
    fig, axes = mngs.plt.subplots(2, 2)
    assert fig is not None
    assert axes.shape == (2, 2)

    # With figure size
    fig, ax = mngs.plt.subplots(figsize=(10, 6))
    assert fig.get_figwidth() == 10
    assert fig.get_figheight() == 6

    # With shared axes
    fig, axes = mngs.plt.subplots(2, 1, sharex=True)
    assert len(axes) == 2


def test_plot_types():
    """Test various plot types available in mngs.plt."""
    import mngs.plt

    fig, ax = mngs.plt.subplots()
    x = np.linspace(0, 10, 50)
    y = np.sin(x)
    
    # Line plot variations
    ax.plot(x, y, 'r-')
    ax.plot(x, y * 0.5, 'b--', linewidth=2)
    ax.plot(x, y * 0.3, 'g:', alpha=0.5)
    
    # Scatter variations
    ax.scatter(x[::5], y[::5], c='red', s=50)
    ax.scatter(x[::5], -y[::5], marker='^', edgecolors='black')
    
    # Bar variations
    ax.bar([1, 2, 3], [1, 2, 3], width=0.5)
    ax.barh([4, 5, 6], [1, 2, 3], height=0.5)
    
    # Histogram
    ax.hist(np.random.randn(100), bins=20)
    
    # Error bars
    ax.errorbar(x[::10], y[::10], yerr=0.1, fmt='o')
    
    # Fill between
    ax.fill_between(x, y-0.1, y+0.1, alpha=0.3)
    
    # All should complete without error
    assert True


def test_axis_customization():
    """Test axis customization features."""
    import mngs.plt

    fig, ax = mngs.plt.subplots()
    
    # Limits
    ax.set_xlim(0, 10)
    ax.set_ylim(-2, 2)
    assert ax.get_xlim() == (0, 10)
    assert ax.get_ylim() == (-2, 2)
    
    # Scales
    ax.set_xscale('log')
    ax.set_yscale('linear')
    
    # Ticks
    ax.set_xticks([1, 2, 5, 10])
    ax.set_xticklabels(['1', '2', '5', '10'])
    
    # Grid
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Twin axes
    ax2 = ax.twinx()
    assert ax2 is not None


def test_text_and_annotations():
    """Test text and annotation features."""
    import mngs.plt

    fig, ax = mngs.plt.subplots()
    
    # Text
    ax.text(0.5, 0.5, 'Center Text', ha='center', va='center')
    
    # Annotation
    ax.annotate('Point', xy=(0.3, 0.3), xytext=(0.5, 0.7),
                arrowprops=dict(arrowstyle='->', color='red'))
    
    # Title with properties
    ax.set_title('Complex Title', fontsize=16, fontweight='bold', color='blue')
    
    # Axis labels with properties
    ax.set_xlabel('X Axis', fontsize=14, style='italic')
    ax.set_ylabel('Y Axis', fontsize=14, rotation=45)
    
    # Legend
    ax.plot([1, 2], [1, 2], label='Line 1')
    ax.plot([1, 2], [2, 1], label='Line 2')
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)


def test_color_and_style():
    """Test color and style features."""
    import mngs.plt

    fig, ax = mngs.plt.subplots()
    
    # Color formats
    x = np.linspace(0, 10, 100)
    ax.plot(x, np.sin(x), color='red')
    ax.plot(x, np.sin(x + 1), color='#00FF00')
    ax.plot(x, np.sin(x + 2), color=(0, 0, 1))
    ax.plot(x, np.sin(x + 3), color=(0, 0, 0, 0.5))  # RGBA
    
    # Line styles
    ax.plot(x, np.cos(x), linestyle='-')
    ax.plot(x, np.cos(x + 1), linestyle='--')
    ax.plot(x, np.cos(x + 2), linestyle=':')
    ax.plot(x, np.cos(x + 3), linestyle='-.')
    
    # Markers
    markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', '+', 'x']
    for i, marker in enumerate(markers[:5]):
        ax.plot(x[::20], np.sin(x[::20] + i*0.5), marker=marker, linestyle='')


def test_3d_plotting():
    """Test 3D plotting capabilities if available."""
    import mngs.plt
    
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = mngs.plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # 3D scatter
        xs = np.random.rand(100)
        ys = np.random.rand(100)
        zs = np.random.rand(100)
        ax.scatter(xs, ys, zs)
        
        # 3D line
        theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
        z = np.linspace(-2, 2, 100)
        r = z**2 + 1
        x = r * np.sin(theta)
        y = r * np.cos(theta)
        ax.plot(x, y, z)
        
        # 3D surface
        X = np.arange(-5, 5, 0.25)
        Y = np.arange(-5, 5, 0.25)
        X, Y = np.meshgrid(X, Y)
        R = np.sqrt(X**2 + Y**2)
        Z = np.sin(R)
        ax.plot_surface(X, Y, Z)
        
        assert True
    except ImportError:
        pytest.skip("3D plotting not available")


def test_image_handling():
    """Test image display and manipulation."""
    import mngs.plt

    fig, ax = mngs.plt.subplots()
    
    # Create test image
    img = np.random.rand(100, 100)
    
    # Display image
    im = ax.imshow(img, cmap='viridis', aspect='auto')
    assert im is not None
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    assert cbar is not None
    
    # Different interpolations
    ax.imshow(img, interpolation='nearest')
    ax.imshow(img, interpolation='bilinear')
    
    # Extent and origin
    ax.imshow(img, extent=[0, 10, 0, 10], origin='lower')


def test_subplot_layouts():
    """Test various subplot layout configurations."""
    import mngs.plt

    # GridSpec
    fig = mngs.plt.figure()
    gs = fig.add_gridspec(3, 3)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :-1])
    ax3 = fig.add_subplot(gs[1:, -1])
    ax4 = fig.add_subplot(gs[-1, 0])
    ax5 = fig.add_subplot(gs[-1, -2])
    
    assert all([ax1, ax2, ax3, ax4, ax5])
    
    # Constrained layout
    fig, axes = mngs.plt.subplots(2, 2, constrained_layout=True)
    assert axes.shape == (2, 2)


def test_save_functionality():
    """Test figure saving capabilities."""
    import mngs.plt
    import tempfile
    import os

    fig, ax = mngs.plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save in different formats
        formats = ['png', 'pdf', 'svg', 'jpg']
        for fmt in formats:
            filepath = os.path.join(tmpdir, f'test.{fmt}')
            fig.savefig(filepath, format=fmt)
            assert os.path.exists(filepath)
        
        # Save with options
        filepath = os.path.join(tmpdir, 'test_dpi.png')
        fig.savefig(filepath, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        assert os.path.exists(filepath)


def test_interactive_features():
    """Test interactive features if available."""
    import mngs.plt

    fig, ax = mngs.plt.subplots()
    
    # Zoom and pan (just test they don't error)
    if hasattr(ax, 'set_navigate_mode'):
        ax.set_navigate_mode('pan')
        ax.set_navigate_mode('zoom')
        ax.set_navigate_mode(None)
    
    # Picker events
    line, = ax.plot([1, 2, 3], [1, 2, 3], picker=True)
    
    # Cursor
    if hasattr(ax, 'format_coord'):
        coord_str = ax.format_coord(1.5, 1.5)
        assert isinstance(coord_str, str)


def test_style_contexts():
    """Test matplotlib style contexts."""
    import mngs.plt

    # Test with style context
    try:
        with plt.style.context('seaborn'):
            fig, ax = mngs.plt.subplots()
            ax.plot([1, 2, 3], [1, 2, 3])
    except:
        # Style might not be available
        pass
    
    # Test rcParams modification
    original_linewidth = plt.rcParams['lines.linewidth']
    plt.rcParams['lines.linewidth'] = 3.0
    
    fig, ax = mngs.plt.subplots()
    line, = ax.plot([1, 2, 3], [1, 2, 3])
    
    # Restore
    plt.rcParams['lines.linewidth'] = original_linewidth


def test_special_plots():
    """Test special plot types."""
    import mngs.plt

    fig, ax = mngs.plt.subplots()
    
    # Pie chart
    ax.pie([1, 2, 3, 4], labels=['A', 'B', 'C', 'D'], autopct='%1.1f%%')
    
    # Box plot
    ax.clear()
    data = [np.random.randn(100) for _ in range(4)]
    ax.boxplot(data)
    
    # Violin plot
    ax.clear()
    ax.violinplot(data)
    
    # Contour plot
    ax.clear()
    X, Y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    Z = np.sin(X) * np.cos(Y)
    ax.contour(X, Y, Z)
    ax.contourf(X, Y, Z, alpha=0.5)


def test_axes_properties():
    """Test axes property getters and setters."""
    import mngs.plt

    fig, ax = mngs.plt.subplots()
    
    # Aspect ratio
    ax.set_aspect('equal')
    ax.set_aspect(2.0)
    ax.set_aspect('auto')
    
    # Background color
    ax.set_facecolor('lightgray')
    
    # Margins
    ax.margins(0.1)
    ax.margins(x=0.2, y=0.3)
    
    # Autoscaling
    ax.autoscale(enable=True, axis='both', tight=True)
    ax.autoscale(enable=False)
    
    # Visibility
    ax.set_visible(False)
    ax.set_visible(True)


def test_fallback_mechanism():
    """Test that mngs.plt can fall back to matplotlib for missing functionality."""
    import warnings

    import mngs.plt
    
    try:
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
                        "mT",  # Skip matrix transpose which causes issues with ndim < 2
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

<<<<<<< HEAD
                        # Attempt to access and call the method
                        # Just check it exists, we won't actually call it as we don't know the parameters
                        method = getattr(ax, method_name, None)
                        assert (
                            method is not None
                        ), f"Method {method_name} should be available via fallback"

                        # Check that a warning was issued
                        assert any(
                            "fallback" in str(w.message).lower() for w in warning_list
                        ), f"No fallback warning issued for {method_name}"
=======
                            # Check that a warning was issued
                            assert any(
                                "fallback" in str(w.message).lower()
                                for w in warning_list
                            ), f"No fallback warning issued for {method_name}"
    except ValueError as e:
        if "matrix transpose with ndim < 2 is undefined" in str(e):
            # Skip test if we encounter the matrix transpose error
            # This occurs in NumPy's masked array implementation
            pytest.skip("Skipping test due to matrix transpose dimensionality issue")
>>>>>>> origin/main


def test_subplots_creation():
    """Test various ways to create subplots."""
    import mngs.plt

    # Single subplot
    fig, ax = mngs.plt.subplots()
    assert fig is not None
    assert ax is not None

    # Multiple subplots
    fig, axes = mngs.plt.subplots(2, 2)
    assert fig is not None
    assert axes.shape == (2, 2)

    # With figure size
    fig, ax = mngs.plt.subplots(figsize=(10, 6))
    assert fig.get_figwidth() == 10
    assert fig.get_figheight() == 6

    # With shared axes
    fig, axes = mngs.plt.subplots(2, 1, sharex=True)
    assert len(axes) == 2


def test_plot_types():
    """Test various plot types available in mngs.plt."""
    import mngs.plt

    fig, ax = mngs.plt.subplots()
    x = np.linspace(0, 10, 50)
    y = np.sin(x)
    
    # Line plot variations
    ax.plot(x, y, 'r-')
    ax.plot(x, y * 0.5, 'b--', linewidth=2)
    ax.plot(x, y * 0.3, 'g:', alpha=0.5)
    
    # Scatter variations
    ax.scatter(x[::5], y[::5], c='red', s=50)
    ax.scatter(x[::5], -y[::5], marker='^', edgecolors='black')
    
    # Bar variations
    ax.bar([1, 2, 3], [1, 2, 3], width=0.5)
    ax.barh([4, 5, 6], [1, 2, 3], height=0.5)
    
    # Histogram
    ax.hist(np.random.randn(100), bins=20)
    
    # Error bars
    ax.errorbar(x[::10], y[::10], yerr=0.1, fmt='o')
    
    # Fill between
    ax.fill_between(x, y-0.1, y+0.1, alpha=0.3)
    
    # All should complete without error
    assert True


def test_axis_customization():
    """Test axis customization features."""
    import mngs.plt

    fig, ax = mngs.plt.subplots()
    
    # Limits
    ax.set_xlim(0, 10)
    ax.set_ylim(-2, 2)
    assert ax.get_xlim() == (0, 10)
    assert ax.get_ylim() == (-2, 2)
    
    # Scales
    ax.set_xscale('log')
    ax.set_yscale('linear')
    
    # Ticks
    ax.set_xticks([1, 2, 5, 10])
    ax.set_xticklabels(['1', '2', '5', '10'])
    
    # Grid
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Twin axes
    ax2 = ax.twinx()
    assert ax2 is not None


def test_text_and_annotations():
    """Test text and annotation features."""
    import mngs.plt

    fig, ax = mngs.plt.subplots()
    
    # Text
    ax.text(0.5, 0.5, 'Center Text', ha='center', va='center')
    
    # Annotation
    ax.annotate('Point', xy=(0.3, 0.3), xytext=(0.5, 0.7),
                arrowprops=dict(arrowstyle='->', color='red'))
    
    # Title with properties
    ax.set_title('Complex Title', fontsize=16, fontweight='bold', color='blue')
    
    # Axis labels with properties
    ax.set_xlabel('X Axis', fontsize=14, style='italic')
    ax.set_ylabel('Y Axis', fontsize=14, rotation=45)
    
    # Legend
    ax.plot([1, 2], [1, 2], label='Line 1')
    ax.plot([1, 2], [2, 1], label='Line 2')
    ax.legend(loc='upper right', frameon=True, fancybox=True)


def test_color_and_style():
    """Test color formats and line styles."""
    import mngs.plt

    fig, ax = mngs.plt.subplots()
    
    # Different color formats
    ax.plot([1, 2, 3], [1, 2, 3], color='red')
    ax.plot([1, 2, 3], [2, 3, 4], color='#FF5733')
    ax.plot([1, 2, 3], [3, 4, 5], color=(0.1, 0.2, 0.5))
    ax.plot([1, 2, 3], [4, 5, 6], color='C3')
    
    # Different line styles
    ax.plot([1, 2, 3], [5, 6, 7], linestyle='-')
    ax.plot([1, 2, 3], [6, 7, 8], linestyle='--')
    ax.plot([1, 2, 3], [7, 8, 9], linestyle=':')
    ax.plot([1, 2, 3], [8, 9, 10], linestyle='-.')
    
    # Markers
    ax.plot([1, 2, 3], [9, 10, 11], marker='o', markersize=10)
    ax.plot([1, 2, 3], [10, 11, 12], marker='s', markerfacecolor='red')


def test_3d_plotting():
    """Test 3D plotting capabilities."""
    import mngs.plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = mngs.plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # 3D scatter
    x = np.random.randn(100)
    y = np.random.randn(100)
    z = np.random.randn(100)
    ax.scatter(x, y, z)
    
    # 3D line
    t = np.linspace(0, 10, 100)
    ax.plot(np.sin(t), np.cos(t), t)
    
    # Surface plot
    X, Y = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
    Z = np.sin(np.sqrt(X**2 + Y**2))
    ax.plot_surface(X, Y, Z)


def test_image_handling():
    """Test image display and manipulation."""
    import mngs.plt

    fig, ax = mngs.plt.subplots()
    
    # Create and display an image
    img = np.random.rand(100, 100)
    im = ax.imshow(img, cmap='viridis', aspect='auto')
    
    # Add colorbar
    fig.colorbar(im, ax=ax)
    
    # Test different interpolations
    ax.imshow(img, interpolation='nearest')
    ax.imshow(img, interpolation='bilinear')
    
    # Test extent and origin
    ax.imshow(img, extent=[0, 10, 0, 5], origin='lower')


def test_subplot_layouts():
    """Test different subplot layouts and GridSpec."""
    import mngs.plt

    # GridSpec
    fig = mngs.plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Different sized subplots
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1:, 0])
    ax3 = fig.add_subplot(gs[1:, 1:])
    
    assert ax1 is not None
    assert ax2 is not None
    assert ax3 is not None
    
    # Subplot2grid
    ax4 = mngs.plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
    assert ax4 is not None


def test_save_functionality():
    """Test figure saving in different formats."""
    import tempfile
    import mngs.plt

    fig, ax = mngs.plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test different formats
        formats = ['png', 'pdf', 'svg', 'jpg']
        for fmt in formats:
            filepath = f"{tmpdir}/test.{fmt}"
            fig.savefig(filepath, dpi=100, bbox_inches='tight')
            assert os.path.exists(filepath)


def test_interactive_features():
    """Test interactive plot features."""
    import mngs.plt

    # Test ion/ioff
    mngs.plt.ion()
    mngs.plt.ioff()
    
    # Test show (in non-blocking mode)
    fig, ax = mngs.plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    # Note: We can't really test plt.show() in automated tests
    
    # Test pause
    mngs.plt.pause(0.001)
    
    # Test draw
    fig.canvas.draw()


def test_style_contexts():
    """Test matplotlib style contexts."""
    import mngs.plt

    # Test style context - use a built-in style that's always available
    with mngs.plt.style.context('default'):
        fig, ax = mngs.plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
    
    # Test rc params
    mngs.plt.rc('lines', linewidth=2, linestyle='--')
    mngs.plt.rcdefaults()
    
    # Test rcParams
    old_linewidth = mngs.plt.rcParams['lines.linewidth']
    mngs.plt.rcParams['lines.linewidth'] = 3
    mngs.plt.rcParams['lines.linewidth'] = old_linewidth


def test_special_plots():
    """Test special plot types."""
    import mngs.plt

    fig, axes = mngs.plt.subplots(2, 2, figsize=(10, 10))
    
    # Pie chart
    axes[0, 0].pie([30, 25, 20, 25], labels=['A', 'B', 'C', 'D'])
    
    # Box plot
    data = [np.random.randn(100) for _ in range(4)]
    axes[0, 1].boxplot(data)
    
    # Violin plot
    axes[1, 0].violinplot(data)
    
    # Contour plot
    X, Y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    Z = np.sin(X) * np.cos(Y)
    axes[1, 1].contour(X, Y, Z)


def test_axes_properties():
    """Test axes property getters and setters."""
    import mngs.plt

    fig, ax = mngs.plt.subplots()
    
    # Test aspect ratio
    ax.set_aspect('equal')
    ax.set_aspect(2.0)
    ax.set_aspect('auto')
    
    # Test axis on/off
    ax.axis('off')
    ax.axis('on')
    
    # Test tight layout
    fig.tight_layout()
    
    # Test visibility
    ax.set_visible(False)
    ax.set_visible(True)


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

    # Generate and export the report when run directly
    report = generate_comprehensive_report()
    export_report_to_markdown(report, os.path.join(__DIR__, "compatibility_report.md"))
    print(f"Report exported to {os.path.join(__DIR__, 'compatibility_report.md')}")

# EOF
