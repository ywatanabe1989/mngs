#!/usr/bin/env python3
# -*- coding: utf-8 -*-
<<<<<<< HEAD
# Timestamp: "2025-06-12 13:45:00 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/.claude-worktree/mngs_repo/src/mngs/io/_save_modules/_mp4.py
# ----------------------------------------
import os

__FILE__ = "./src/mngs/io/_save_modules/_mp4.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
MP4 video file saving functionality for mngs.io.save

This module acts as a wrapper around the existing _save_mp4.py functionality
to maintain consistency with the modular save system.
"""

from ._save_mp4 import _mk_mp4


def save_mp4(obj, spath, **kwargs):
    """Handle MP4 video file saving.
    
    Parameters
    ----------
    obj : list of images or array-like
        List of images or 3D/4D array to save as MP4 video
    spath : str
        Path where MP4 file will be saved
    **kwargs
        Additional keyword arguments passed to _mk_mp4()
        Common kwargs: fps (frames per second), codec, bitrate
        
    Notes
    -----
    This is a wrapper around the existing _mk_mp4 function
    """
    _mk_mp4(obj, spath, **kwargs)


# EOF
=======
# Time-stamp: "2024-11-02 16:57:29 (ywatanabe)"
# File: ./mngs_repo/src/mngs/io/_save_mp4.py

from matplotlib import animation


def _mk_mp4(fig, spath_mp4):
    axes = fig.get_axes()

    def init():
        return (fig,)

    def animate(i):
        for ax in axes:
            ax.view_init(elev=10.0, azim=i)
        return (fig,)

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=360, interval=20, blit=True
    )

    writermp4 = animation.FFMpegWriter(fps=60, extra_args=["-vcodec", "libx264"])
    anim.save(spath_mp4, writer=writermp4)
    print("\nSaving to: {}\n".format(spath_mp4))


# EOF
>>>>>>> origin/main
