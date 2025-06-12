#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-31 20:05:00 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/.claude-worktree/mngs_repo/tests/mngs/io/test__save_mp4.py

import pytest
import os
import tempfile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from unittest.mock import patch, MagicMock, call
from matplotlib import animation
import numpy as np
from mngs.io._save_mp4 import _mk_mp4


class TestMkMp4:
    """Test cases for _mk_mp4 function."""

    @pytest.fixture
    def sample_3d_figure(self):
        """Create a sample 3D matplotlib figure."""
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        # Create some sample 3D data
        x = np.random.rand(100)
        y = np.random.rand(100)
        z = np.random.rand(100)
        ax.scatter(x, y, z)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        yield fig
        plt.close(fig)

    @pytest.fixture
    def sample_2d_figure(self):
        """Create a sample 2D matplotlib figure (no 3D axes)."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        yield fig
        plt.close(fig)

    @patch("mngs.io._save_mp4.animation.FFMpegWriter")
    @patch("mngs.io._save_mp4.animation.FuncAnimation")
    def test_mk_mp4_basic(self, mock_anim, mock_writer, sample_3d_figure, capsys):
        """Test basic MP4 creation with 3D figure."""
        # Setup mocks
        mock_writer_instance = MagicMock()
        mock_writer.return_value = mock_writer_instance
        mock_anim_instance = MagicMock()
        mock_anim.return_value = mock_anim_instance

        # Create temp file path
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            temp_path = tmp.name

        try:
            # Call function
            _mk_mp4(sample_3d_figure, temp_path)

            # Verify FuncAnimation was called correctly
            mock_anim.assert_called_once()
            call_args = mock_anim.call_args
            assert call_args[0][0] == sample_3d_figure  # fig
            assert call_args[1]["frames"] == 360
            assert call_args[1]["interval"] == 20
            assert call_args[1]["blit"] == True

            # Verify FFMpegWriter was configured
            mock_writer.assert_called_once_with(
                fps=60, extra_args=["-vcodec", "libx264"]
            )

            # Verify save was called
            mock_anim_instance.save.assert_called_once_with(
                temp_path, writer=mock_writer_instance
            )

            # Check output message
            captured = capsys.readouterr()
            assert f"Saving to: {temp_path}" in captured.out

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @patch("mngs.io._save_mp4.animation.FuncAnimation")
    def test_mk_mp4_animation_callbacks(self, mock_anim, sample_3d_figure):
        """Test that animation callbacks work correctly."""
        # Capture the animation functions
        init_func = None
        animate_func = None

        def capture_funcs(*args, **kwargs):
            nonlocal init_func, animate_func
            init_func = kwargs.get("init_func")
            animate_func = args[1]  # Second positional argument
            return MagicMock()

        mock_anim.side_effect = capture_funcs

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            temp_path = tmp.name

        try:
            # Call function to capture callbacks
            _mk_mp4(sample_3d_figure, temp_path)

            # Test init function
            result = init_func()
            assert result == (sample_3d_figure,)

            # Test animate function
            ax = sample_3d_figure.get_axes()[0]
            initial_azim = ax.azim

            # Call animate with frame 90
            result = animate_func(90)
            assert result == (sample_3d_figure,)

            # Check that view was updated
            assert ax.azim == 90
            assert ax.elev == 10.0

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @patch("mngs.io._save_mp4.animation.FFMpegWriter")
    @patch("mngs.io._save_mp4.animation.FuncAnimation")
    def test_mk_mp4_with_2d_figure(self, mock_anim, mock_writer, sample_2d_figure):
        """Test MP4 creation with 2D figure (axes without view_init)."""
        mock_writer_instance = MagicMock()
        mock_writer.return_value = mock_writer_instance
        mock_anim_instance = MagicMock()
        mock_anim.return_value = mock_anim_instance

        # Capture animate function
        animate_func = None

        def capture_animate(*args, **kwargs):
            nonlocal animate_func
            animate_func = args[1]
            return mock_anim_instance

        mock_anim.side_effect = capture_animate

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            temp_path = tmp.name

        try:
            _mk_mp4(sample_2d_figure, temp_path)

            # Test that animate function handles 2D axes gracefully
            # (they don't have view_init method)
            result = animate_func(45)
            assert result == (sample_2d_figure,)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @patch("mngs.io._save_mp4.animation.FFMpegWriter")
    @patch("mngs.io._save_mp4.animation.FuncAnimation")
    def test_mk_mp4_multiple_axes(self, mock_anim, mock_writer):
        """Test MP4 creation with figure containing multiple 3D axes."""
        # Create figure with multiple 3D subplots
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")

        # Add some data
        x = np.random.rand(50)
        y = np.random.rand(50)
        z = np.random.rand(50)
        ax1.scatter(x, y, z)
        ax2.plot(x, y, z)

        # Capture animate function
        animate_func = None

        def capture_animate(*args, **kwargs):
            nonlocal animate_func
            animate_func = args[1]
            return MagicMock()

        mock_anim.side_effect = capture_animate

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            temp_path = tmp.name

        try:
            _mk_mp4(fig, temp_path)

            # Test that animate updates both axes
            animate_func(180)

            # Both axes should have azim=180
            assert ax1.azim == 180
            assert ax2.azim == 180
            assert ax1.elev == 10.0
            assert ax2.elev == 10.0

        finally:
            plt.close(fig)
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @patch("mngs.io._save_mp4.animation.FFMpegWriter")
    def test_mk_mp4_save_error(self, mock_writer, sample_3d_figure):
        """Test error handling during save."""
        mock_writer_instance = MagicMock()
        mock_writer.return_value = mock_writer_instance

        # Make save raise an exception
        with patch("mngs.io._save_mp4.animation.FuncAnimation") as mock_anim:
            mock_anim_instance = MagicMock()
            mock_anim.return_value = mock_anim_instance
            mock_anim_instance.save.side_effect = Exception("FFmpeg not found")

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                temp_path = tmp.name

            try:
                # Should raise the exception
                with pytest.raises(Exception, match="FFmpeg not found"):
                    _mk_mp4(sample_3d_figure, temp_path)
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

    def test_mk_mp4_full_rotation(self, sample_3d_figure):
        """Test that animation creates full 360 degree rotation."""
        # This test verifies the animation logic without mocking
        with patch("mngs.io._save_mp4.animation.FuncAnimation") as mock_anim:
            # Capture animation parameters
            captured_frames = None

            def capture_params(*args, **kwargs):
                nonlocal captured_frames
                captured_frames = kwargs.get("frames", 0)
                return MagicMock()

            mock_anim.side_effect = capture_params

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                temp_path = tmp.name

            try:
                _mk_mp4(sample_3d_figure, temp_path)

                # Verify 360 frames for full rotation
                assert captured_frames == 360

            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)


if __name__ == "__main__":
    import os
    import pytest

    pytest.main([os.path.abspath(__file__)])
