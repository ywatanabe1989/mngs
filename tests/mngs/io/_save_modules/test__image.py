#!/usr/bin/env python3
# -*- coding: utf-8 -*-
<<<<<<< HEAD
# Timestamp: "2025-06-12 13:57:00 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/.claude-worktree/mngs_repo/tests/mngs/io/_save_modules/test__image.py
# ----------------------------------------
import os

__FILE__ = "./tests/mngs/io/_save_modules/test__image.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Test cases for image saving wrapper functionality
"""

import os
import tempfile
import pytest
import numpy as np
from pathlib import Path

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from mngs.io._save_modules._image import save_image


class TestSaveImage:
    """Test suite for save_image wrapper function"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file_png = os.path.join(self.temp_dir, "test.png")
        self.test_file_jpg = os.path.join(self.temp_dir, "test.jpg")

    def teardown_method(self):
        """Clean up after tests"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        # Close any matplotlib figures
        if MATPLOTLIB_AVAILABLE:
            plt.close('all')

    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL/Pillow not installed")
    def test_save_pil_image(self):
        """Test saving PIL Image object"""
        # Create a simple RGB image
        img = Image.new('RGB', (100, 100), color='red')
        save_image(img, self.test_file_png)
        
        assert os.path.exists(self.test_file_png)
        
        # Verify saved image
        loaded = Image.open(self.test_file_png)
        assert loaded.size == (100, 100)
        assert loaded.mode == 'RGB'

    def test_save_numpy_array_rgb(self):
        """Test saving numpy array as image (RGB)"""
        # Create RGB image array (H, W, C)
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[:, :, 0] = 255  # Red channel
        
        save_image(arr, self.test_file_png)
        
        assert os.path.exists(self.test_file_png)
        
        if PIL_AVAILABLE:
            loaded = Image.open(self.test_file_png)
            assert loaded.size == (100, 100)

    def test_save_numpy_array_grayscale(self):
        """Test saving grayscale numpy array"""
        # Create grayscale image
        arr = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
        save_image(arr, self.test_file_png)
        
        assert os.path.exists(self.test_file_png)

    def test_save_numpy_array_float(self):
        """Test saving float numpy array (0-1 range)"""
        # Create float array
        arr = np.random.rand(100, 100, 3)
        
        save_image(arr, self.test_file_png)
        
        assert os.path.exists(self.test_file_png)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not installed")
    def test_save_matplotlib_figure(self):
        """Test saving matplotlib figure"""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
        ax.set_title("Test Plot")
        
        save_image(fig, self.test_file_png)
        
        assert os.path.exists(self.test_file_png)
        plt.close(fig)

    def test_save_different_formats(self):
        """Test saving in different image formats"""
        arr = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        
        # PNG
        save_image(arr, self.test_file_png)
        assert os.path.exists(self.test_file_png)
        
        # JPEG
        save_image(arr, self.test_file_jpg)
        assert os.path.exists(self.test_file_jpg)
        
        # BMP
        bmp_file = os.path.join(self.temp_dir, "test.bmp")
        save_image(arr, bmp_file)
        assert os.path.exists(bmp_file)

    def test_save_with_quality(self):
        """Test saving JPEG with quality setting"""
        arr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # High quality
        high_quality = os.path.join(self.temp_dir, "high_quality.jpg")
        save_image(arr, high_quality, quality=95)
        
        # Low quality
        low_quality = os.path.join(self.temp_dir, "low_quality.jpg")
        save_image(arr, low_quality, quality=10)
        
        # High quality file should be larger
        assert os.path.getsize(high_quality) > os.path.getsize(low_quality)

    def test_save_rgba_image(self):
        """Test saving image with alpha channel"""
        # Create RGBA image
        arr = np.zeros((100, 100, 4), dtype=np.uint8)
        arr[:, :, 0] = 255  # Red
        arr[:, :, 3] = 128  # Semi-transparent
        
        save_image(arr, self.test_file_png)
        
        assert os.path.exists(self.test_file_png)
        
        if PIL_AVAILABLE:
            loaded = Image.open(self.test_file_png)
            assert loaded.mode in ['RGBA', 'LA']

    def test_save_large_image(self):
        """Test saving large image"""
        # Create large image
        arr = np.random.randint(0, 256, (2000, 2000, 3), dtype=np.uint8)
        
        save_image(arr, self.test_file_png)
        
        assert os.path.exists(self.test_file_png)

    def test_save_binary_image(self):
        """Test saving binary (black and white) image"""
        # Create binary image
        arr = np.random.choice([0, 255], size=(100, 100), p=[0.5, 0.5]).astype(np.uint8)
        
        save_image(arr, self.test_file_png)
        
        assert os.path.exists(self.test_file_png)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not installed")
    def test_save_matplotlib_with_dpi(self):
        """Test saving matplotlib figure with custom DPI"""
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([1, 2, 3], [1, 2, 3])
        
        # Save with high DPI
        save_image(fig, self.test_file_png, dpi=300)
        
        assert os.path.exists(self.test_file_png)
        plt.close(fig)

    def test_save_from_file_path(self):
        """Test copying existing image file"""
        # First create an image
        arr = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        temp_source = os.path.join(self.temp_dir, "source.png")
        
        if PIL_AVAILABLE:
            Image.fromarray(arr).save(temp_source)
            
            # Now use save_image to copy it
            save_image(temp_source, self.test_file_png)
            
            assert os.path.exists(self.test_file_png)

    def test_error_invalid_input(self):
        """Test error handling for invalid input"""
        with pytest.raises(ValueError):
            save_image("not an image", self.test_file_png)
        
        with pytest.raises(ValueError):
            save_image(123, self.test_file_png)

    def test_save_with_metadata(self):
        """Test saving image with metadata (if supported)"""
        if PIL_AVAILABLE:
            img = Image.new('RGB', (100, 100), color='blue')
            
            # PIL images can have info dict
            img.info['description'] = 'Test image'
            img.info['software'] = 'mngs'
            
            save_image(img, self.test_file_png)
            
            loaded = Image.open(self.test_file_png)
            # Note: Not all formats preserve metadata

    def test_save_palette_image(self):
        """Test saving palette/indexed color image"""
        if PIL_AVAILABLE:
            # Create palette image
            img = Image.new('P', (100, 100))
            palette = []
            for i in range(256):
                palette.extend([i, 0, 0])  # Red gradient
            img.putpalette(palette)
            
            save_image(img, self.test_file_png)
            
            assert os.path.exists(self.test_file_png)
            loaded = Image.open(self.test_file_png)
            assert loaded.mode == 'P'


# EOF
=======
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
<<<<<<<< HEAD:tests/mngs/io/test__save_image.py
========

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_image.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-04-29 19:42:31 (ywatanabe)"
# # File: /home/ywatanabe/proj/mngs_repo/src/mngs/io/_save_image.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/io/_save_image.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# import io as _io
# 
# import plotly
# from PIL import Image
# 
# 
# def _save_image(obj, spath, **kwargs):
#     # try:
#     #     import mngs
# 
#     #     type(obj) == mngs.plt._subplots._FigWrapper.FigWrapper
#     #     obj._called_from_mng_io_save = True
#     # except:
#     #     pass
# 
#     # png
#     if spath.endswith(".png"):
#         # plotly
#         if isinstance(obj, plotly.graph_objs.Figure):
#             obj.write_image(file=spath, format="png")
#         # PIL image
#         elif isinstance(obj, Image.Image):
#             obj.save(spath)
#         # matplotlib
#         else:
#             try:
#                 obj.savefig(spath)
#             except:
#                 obj.figure.savefig(spath)
#         del obj
# 
#     # tiff
#     elif spath.endswith(".tiff") or spath.endswith(".tif"):
#         # PIL image
#         if isinstance(obj, Image.Image):
#             obj.save(spath)
#         # matplotlib
#         else:
#             try:
#                 obj.savefig(spath, dpi=300, format="tiff")
#             except:
#                 obj.figure.savefig(spath, dpi=300, format="tiff")
# 
#         del obj
# 
#     # jpeg
#     elif spath.endswith(".jpeg") or spath.endswith(".jpg"):
#         buf = _io.BytesIO()
# 
#         # plotly
#         if isinstance(obj, plotly.graph_objs.Figure):
#             obj.write_image(buf, format="png")
#             buf.seek(0)
#             img = Image.open(buf)
#             img.convert("RGB").save(spath, "JPEG")
#             buf.close()
# 
#         # PIL image
#         elif isinstance(obj, Image.Image):
#             obj.save(spath)
# 
#         # matplotlib
#         else:
#             try:
#                 obj.savefig(buf, format="png")
#             except:
#                 obj.figure.savefig(buf, format="png")
# 
#             buf.seek(0)
#             img = Image.open(buf)
#             img.convert("RGB").save(spath, "JPEG")
#             buf.close()
#         del obj
# 
#     # SVG
#     elif spath.endswith(".svg"):
#         # Plotly
#         if isinstance(obj, plotly.graph_objs.Figure):
#             obj.write_image(file=spath, format="svg")
#         # Matplotlib
#         else:
#             try:
#                 obj.savefig(spath, format="svg")
#             except AttributeError:
#                 obj.figure.savefig(spath, format="svg")
#         del obj
# 
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_image.py
# --------------------------------------------------------------------------------
>>>>>>>> origin/main:tests/mngs/io/_save_modules/test__image.py
>>>>>>> origin/main
