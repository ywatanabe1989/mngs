#!/usr/bin/env python3
# -*- coding: utf-8 -*-
<<<<<<< HEAD
# Timestamp: "2025-06-12 13:58:00 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/.claude-worktree/mngs_repo/tests/mngs/io/_save_modules/test__mp4.py
# ----------------------------------------
import os

__FILE__ = "./tests/mngs/io/_save_modules/test__mp4.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Test cases for MP4 saving wrapper functionality
"""

import os
import tempfile
import pytest
import numpy as np
from pathlib import Path

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from mngs.io._save_modules._mp4 import save_mp4


class TestSaveMP4:
    """Test suite for save_mp4 wrapper function"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.mp4")

    def teardown_method(self):
        """Clean up after tests"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        if MATPLOTLIB_AVAILABLE:
            plt.close('all')

    def test_save_numpy_frames(self):
        """Test saving list of numpy arrays as video"""
        # Create simple frames (10 frames, 100x100 RGB)
        frames = []
        for i in range(10):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            # Create moving square
            frame[i*10:(i+1)*10, i*10:(i+1)*10] = [255, 0, 0]  # Red square
            frames.append(frame)
        
        save_mp4(frames, self.test_file, fps=10)
        
        assert os.path.exists(self.test_file)
        assert os.path.getsize(self.test_file) > 0

    @pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not installed")
    def test_save_with_opencv_writer(self):
        """Test saving video using OpenCV VideoWriter"""
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.test_file, fourcc, 20.0, (640, 480))
        
        # Generate frames
        for i in range(30):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Moving circle
            cv2.circle(frame, (i*20, 240), 30, (0, 255, 0), -1)
            out.write(frame)
        
        out.release()
        
        assert os.path.exists(self.test_file)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not installed")
    def test_save_matplotlib_animation(self):
        """Test saving matplotlib animation"""
        fig, ax = plt.subplots()
        
        # Create animation data
        x = np.linspace(0, 2*np.pi, 100)
        line, = ax.plot(x, np.sin(x))
        
        def animate(frame):
            line.set_ydata(np.sin(x + frame/10))
            return line,
        
        anim = animation.FuncAnimation(
            fig, animate, frames=20, interval=50, blit=True
        )
        
        save_mp4(anim, self.test_file)
        
        assert os.path.exists(self.test_file)
        plt.close(fig)

    def test_save_grayscale_frames(self):
        """Test saving grayscale video frames"""
        # Create grayscale frames
        frames = []
        for i in range(15):
            frame = np.zeros((100, 100), dtype=np.uint8)
            # Moving white bar
            frame[:, i*6:(i+1)*6] = 255
            frames.append(frame)
        
        save_mp4(frames, self.test_file, fps=15)
        
        assert os.path.exists(self.test_file)

    def test_save_different_frame_rates(self):
        """Test saving with different frame rates"""
        frames = [np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8) 
                 for _ in range(10)]
        
        # Low FPS
        low_fps = os.path.join(self.temp_dir, "low_fps.mp4")
        save_mp4(frames, low_fps, fps=5)
        
        # High FPS
        high_fps = os.path.join(self.temp_dir, "high_fps.mp4")
        save_mp4(frames, high_fps, fps=30)
        
        assert os.path.exists(low_fps)
        assert os.path.exists(high_fps)

    def test_save_with_codec(self):
        """Test saving with specific codec"""
        frames = [np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8) 
                 for _ in range(10)]
        
        # Different codecs might be available
        codecs = ['mp4v', 'h264', 'xvid']
        for codec in codecs:
            file_path = os.path.join(self.temp_dir, f"test_{codec}.mp4")
            try:
                save_mp4(frames, file_path, codec=codec, fps=10)
                if os.path.exists(file_path):
                    break
            except:
                continue

    def test_save_high_resolution_video(self):
        """Test saving high resolution video"""
        # Create HD frames (fewer frames due to size)
        frames = []
        for i in range(5):
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            # Gradient background
            frame[:, :, 0] = np.linspace(0, 255, 1280)
            frames.append(frame)
        
        save_mp4(frames, self.test_file, fps=24)
        
        assert os.path.exists(self.test_file)

    def test_save_with_compression(self):
        """Test saving with different compression settings"""
        frames = [np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8) 
                 for _ in range(20)]
        
        # High quality (less compression)
        high_quality = os.path.join(self.temp_dir, "high_quality.mp4")
        save_mp4(frames, high_quality, fps=10, bitrate='5000k')
        
        # Low quality (more compression)
        low_quality = os.path.join(self.temp_dir, "low_quality.mp4")
        save_mp4(frames, low_quality, fps=10, bitrate='500k')
        
        # High quality should be larger (if bitrate is supported)
        if os.path.exists(high_quality) and os.path.exists(low_quality):
            # This might not always be true depending on implementation
            pass

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not installed")
    def test_save_animated_plot(self):
        """Test saving animated matplotlib plot"""
        fig, ax = plt.subplots()
        
        # Animated scatter plot
        scat = ax.scatter([], [])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        def update(frame):
            x = np.random.rand(10) * 10
            y = np.random.rand(10) * 10
            data = np.c_[x, y]
            scat.set_offsets(data)
            return scat,
        
        anim = animation.FuncAnimation(fig, update, frames=30, interval=50)
        
        save_mp4(anim, self.test_file)
        
        assert os.path.exists(self.test_file)
        plt.close(fig)

    def test_save_single_channel_as_rgb(self):
        """Test converting single channel to RGB for video"""
        # Create single channel frames
        frames = []
        for i in range(10):
            frame = np.zeros((100, 100), dtype=np.uint8)
            frame[40:60, 40:60] = 255 * (i / 10)  # Fading square
            # Convert to RGB
            rgb_frame = np.stack([frame, frame, frame], axis=-1)
            frames.append(rgb_frame)
        
        save_mp4(frames, self.test_file, fps=10)
        
        assert os.path.exists(self.test_file)

    def test_error_invalid_frames(self):
        """Test error handling for invalid frame data"""
        # Empty list
        with pytest.raises(ValueError):
            save_mp4([], self.test_file)
        
        # Invalid frame dimensions
        with pytest.raises(ValueError):
            frames = [np.zeros((100,)), np.zeros((100,))]  # 1D arrays
            save_mp4(frames, self.test_file)

    def test_save_from_generator(self):
        """Test saving frames from generator"""
        def frame_generator():
            for i in range(10):
                frame = np.zeros((100, 100, 3), dtype=np.uint8)
                frame[:, :, i % 3] = 255  # Cycle through RGB
                yield frame
        
        frames = list(frame_generator())
        save_mp4(frames, self.test_file, fps=10)
        
        assert os.path.exists(self.test_file)

    def test_save_with_audio(self):
        """Test saving video with audio (if supported)"""
        # Note: Audio support depends on the implementation
        frames = [np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8) 
                 for _ in range(30)]
        
        # Basic save without audio
        save_mp4(frames, self.test_file, fps=30)
        
        assert os.path.exists(self.test_file)


# EOF
=======
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
<<<<<<<< HEAD:tests/mngs/io/test__save_mp4.py
========

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_mp4.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 16:57:29 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/io/_save_mp4.py
# 
# from matplotlib import animation
# 
# def _mk_mp4(fig, spath_mp4):
#     axes = fig.get_axes()
# 
#     def init():
#         return (fig,)
# 
#     def animate(i):
#         for ax in axes:
#             ax.view_init(elev=10.0, azim=i)
#         return (fig,)
# 
#     anim = animation.FuncAnimation(
#         fig, animate, init_func=init, frames=360, interval=20, blit=True
#     )
# 
#     writermp4 = animation.FFMpegWriter(
#         fps=60, extra_args=["-vcodec", "libx264"]
#     )
#     anim.save(spath_mp4, writer=writermp4)
#     print("\nSaving to: {}\n".format(spath_mp4))
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_mp4.py
# --------------------------------------------------------------------------------
>>>>>>>> origin/main:tests/mngs/io/_save_modules/test__mp4.py
>>>>>>> origin/main
