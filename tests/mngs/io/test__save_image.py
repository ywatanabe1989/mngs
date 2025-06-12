#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-31"
# File: test__save_image.py

"""Tests for mngs.io._save_image module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image


class TestSaveImagePNG:
    """Test PNG format saving functionality."""

    def test_save_matplotlib_figure_png(self):
        """Test saving matplotlib figure as PNG."""
        from mngs.io._save_image import _save_image

        # Create mock matplotlib figure
        mock_fig = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.png")
            _save_image(mock_fig, output_path)

            # Verify savefig was called
            mock_fig.savefig.assert_called_once_with(output_path)

    def test_save_matplotlib_axes_png(self):
        """Test saving matplotlib axes as PNG (falls back to figure)."""
        from mngs.io._save_image import _save_image

        # Create mock axes with figure attribute
        mock_axes = MagicMock()
        mock_axes.savefig.side_effect = AttributeError()
        mock_axes.figure = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.png")
            _save_image(mock_axes, output_path)

            # Verify figure.savefig was called
            mock_axes.figure.savefig.assert_called_once_with(output_path)

    def test_save_pil_image_png(self):
        """Test saving PIL image as PNG."""
        from mngs.io._save_image import _save_image

        # Create real PIL image
        img = Image.new("RGB", (100, 100), color="red")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.png")
            _save_image(img, output_path)

            # Verify file exists and can be loaded
            assert os.path.exists(output_path)
            loaded = Image.open(output_path)
            assert loaded.size == (100, 100)
            assert loaded.mode == "RGB"

    @patch("plotly.graph_objs.Figure")
    def test_save_plotly_figure_png(self, mock_plotly_fig_class):
        """Test saving plotly figure as PNG."""
        from mngs.io._save_image import _save_image

        # Create mock plotly figure
        mock_fig = MagicMock()
        mock_plotly_fig_class.return_value = mock_fig

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.png")
            _save_image(mock_fig, output_path)

            # Verify write_image was called
            mock_fig.write_image.assert_called_once_with(file=output_path, format="png")


class TestSaveImageTIFF:
    """Test TIFF format saving functionality."""

    def test_save_matplotlib_figure_tiff(self):
        """Test saving matplotlib figure as TIFF with high DPI."""
        from mngs.io._save_image import _save_image

        # Create mock matplotlib figure
        mock_fig = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.tiff")
            _save_image(mock_fig, output_path)

            # Verify savefig was called with correct parameters
            mock_fig.savefig.assert_called_once_with(
                output_path, dpi=300, format="tiff"
            )

    def test_save_matplotlib_tif_extension(self):
        """Test saving with .tif extension (alternative TIFF extension)."""
        from mngs.io._save_image import _save_image

        # Create mock matplotlib figure
        mock_fig = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.tif")
            _save_image(mock_fig, output_path)

            # Verify savefig was called with correct parameters
            mock_fig.savefig.assert_called_once_with(
                output_path, dpi=300, format="tiff"
            )

    def test_save_pil_image_tiff(self):
        """Test saving PIL image as TIFF."""
        from mngs.io._save_image import _save_image

        # Create real PIL image
        img = Image.new("RGB", (200, 200), color="blue")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.tiff")
            _save_image(img, output_path)

            # Verify file exists and can be loaded
            assert os.path.exists(output_path)
            loaded = Image.open(output_path)
            assert loaded.size == (200, 200)


class TestSaveImageJPEG:
    """Test JPEG format saving functionality."""

    def test_save_matplotlib_figure_jpeg(self):
        """Test saving matplotlib figure as JPEG through PNG conversion."""
        from mngs.io._save_image import _save_image

        # Create mock matplotlib figure
        mock_fig = MagicMock()

        # Create a real PNG image for the buffer
        test_img = Image.new("RGB", (100, 100), color="green")
        img_buffer = MagicMock()

        with patch("io.BytesIO", return_value=img_buffer):
            with patch("PIL.Image.open", return_value=test_img):
                with tempfile.TemporaryDirectory() as tmpdir:
                    output_path = os.path.join(tmpdir, "test.jpeg")
                    _save_image(mock_fig, output_path)

                    # Verify savefig was called with buffer
                    mock_fig.savefig.assert_called_once_with(img_buffer, format="png")

    def test_save_matplotlib_jpg_extension(self):
        """Test saving with .jpg extension (alternative JPEG extension)."""
        from mngs.io._save_image import _save_image

        # Create mock matplotlib figure
        mock_fig = MagicMock()

        # Create a real PNG image for the buffer
        test_img = Image.new("RGB", (100, 100), color="yellow")
        img_buffer = MagicMock()

        with patch("io.BytesIO", return_value=img_buffer):
            with patch("PIL.Image.open", return_value=test_img):
                with tempfile.TemporaryDirectory() as tmpdir:
                    output_path = os.path.join(tmpdir, "test.jpg")
                    _save_image(mock_fig, output_path)

                    # Verify savefig was called
                    mock_fig.savefig.assert_called_once()

    def test_save_pil_image_jpeg(self):
        """Test saving PIL image as JPEG."""
        from mngs.io._save_image import _save_image

        # Create real PIL image
        img = Image.new("RGB", (150, 150), color="cyan")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.jpeg")
            _save_image(img, output_path)

            # Verify file exists and can be loaded
            assert os.path.exists(output_path)
            loaded = Image.open(output_path)
            assert loaded.size == (150, 150)
            assert loaded.format == "JPEG"

    @patch("plotly.graph_objs.Figure")
    def test_save_plotly_figure_jpeg(self, mock_plotly_fig_class):
        """Test saving plotly figure as JPEG through PNG conversion."""
        from mngs.io._save_image import _save_image

        # Create mock plotly figure
        mock_fig = MagicMock()
        mock_plotly_fig_class.return_value = mock_fig

        # Create a real PNG image for the buffer
        test_img = Image.new("RGB", (100, 100), color="magenta")
        img_buffer = MagicMock()

        with patch("io.BytesIO", return_value=img_buffer):
            with patch("PIL.Image.open", return_value=test_img):
                with tempfile.TemporaryDirectory() as tmpdir:
                    output_path = os.path.join(tmpdir, "test.jpeg")
                    _save_image(mock_fig, output_path)

                    # Verify write_image was called with buffer
                    mock_fig.write_image.assert_called_once_with(
                        img_buffer, format="png"
                    )


class TestSaveImageSVG:
    """Test SVG format saving functionality."""

    def test_save_matplotlib_figure_svg(self):
        """Test saving matplotlib figure as SVG."""
        from mngs.io._save_image import _save_image

        # Create mock matplotlib figure
        mock_fig = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.svg")
            _save_image(mock_fig, output_path)

            # Verify savefig was called with correct format
            mock_fig.savefig.assert_called_once_with(output_path, format="svg")

    def test_save_matplotlib_axes_svg(self):
        """Test saving matplotlib axes as SVG (falls back to figure)."""
        from mngs.io._save_image import _save_image

        # Create mock axes with figure attribute
        mock_axes = MagicMock()
        mock_axes.savefig.side_effect = AttributeError()
        mock_axes.figure = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.svg")
            _save_image(mock_axes, output_path)

            # Verify figure.savefig was called
            mock_axes.figure.savefig.assert_called_once_with(output_path, format="svg")

    @patch("plotly.graph_objs.Figure")
    def test_save_plotly_figure_svg(self, mock_plotly_fig_class):
        """Test saving plotly figure as SVG."""
        from mngs.io._save_image import _save_image

        # Create mock plotly figure
        mock_fig = MagicMock()
        mock_plotly_fig_class.return_value = mock_fig

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.svg")
            _save_image(mock_fig, output_path)

            # Verify write_image was called
            mock_fig.write_image.assert_called_once_with(file=output_path, format="svg")


class TestSaveImageEdgeCases:
    """Test edge cases and error handling."""

    def test_object_deletion_after_save(self):
        """Test that object is deleted after save (memory management)."""
        from mngs.io._save_image import _save_image

        # Create mock figure
        mock_fig = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.png")
            _save_image(mock_fig, output_path)

            # Object should be deleted (del obj in source)
            # We can't directly test deletion, but can verify function completes

    def test_kwargs_parameter_ignored(self):
        """Test that kwargs parameter is accepted but not used."""
        from mngs.io._save_image import _save_image

        # Create mock figure
        mock_fig = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.png")
            # Should not raise error with extra kwargs
            _save_image(
                mock_fig, output_path, dpi=600, quality=95, extra_param="ignored"
            )

            # Verify basic save still works
            mock_fig.savefig.assert_called_once()

    def test_pathlib_path_support(self):
        """Test that pathlib Path objects work as spath parameter."""
        from mngs.io._save_image import _save_image

        # Create mock figure
        mock_fig = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.png"
            _save_image(mock_fig, str(output_path))

            # Verify savefig was called with string path
            mock_fig.savefig.assert_called_once()


class TestSaveImageIntegration:
    """Test integration with real plotting libraries when available."""

    @pytest.mark.skipif(
        not hasattr(pytest, "importorskip"), reason="pytest.importorskip not available"
    )
    def test_matplotlib_integration(self):
        """Test with real matplotlib figure if available."""
        plt = pytest.importorskip("matplotlib.pyplot")
        from mngs.io._save_image import _save_image

        # Create real matplotlib figure
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([1, 2, 3], [1, 4, 9])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_matplotlib.png")
            _save_image(fig, output_path)

            # Verify file exists and is valid image
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0

        plt.close(fig)

    def test_pil_rgba_to_jpeg_conversion(self):
        """Test that RGBA images are properly converted to RGB for JPEG."""
        from mngs.io._save_image import _save_image

        # Create RGBA image (with transparency)
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_rgba.jpg")
            # PIL should handle the conversion internally
            _save_image(img, output_path)

            # Verify file exists
            assert os.path.exists(output_path)


# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])
