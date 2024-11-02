#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-02 04:01:54 (ywatanabe)"
# File: ./mngs_repo/src/mngs/general/_symlink.py

import os
from ._color_text import color_text

def symlink(tgt, src, force=False):
    """Create a symbolic link.

    This function creates a symbolic link from the target to the source.
    If the force parameter is True, it will remove any existing file at
    the source path before creating the symlink.

    Parameters
    ----------
    tgt : str
        The target path (the file or directory to be linked to).
    src : str
        The source path (where the symbolic link will be created).
    force : bool, optional
        If True, remove the existing file at the src path before creating
        the symlink (default is False).

    Returns
    -------
    None

    Raises
    ------
    OSError
        If the symlink creation fails.

    Example
    -------
    >>> symlink('/path/to/target', '/path/to/link')
    >>> symlink('/path/to/target', '/path/to/existing_file', force=True)
    """
    if force:
        try:
            os.remove(src)
        except FileNotFoundError:
            pass

    # Calculate the relative path from src to tgt
    src_dir = os.path.dirname(src)
    relative_tgt = os.path.relpath(tgt, src_dir)

    os.symlink(relative_tgt, src)
    print(
        color_text(
            f"\nSymlink was created: {src} -> {relative_tgt}\n", c="yellow"
        )
    )


#     os.symlink(tgt, src)
#     print(mngs.gen.ct(f"\nSymlink was created: {src} -> {tgt}\n", c="yellow"))
# Symlink was created: ./scripts/ml/clf/sct_optuna/optuna_studies/optuna_study_stent_3_classes/best_trial -> /home/ywatanabe/proj/ecog_stent_sheep_visual/scripts/ml/clf/sct_optuna/RUNNING/2024Y-03M-29D-21h55m09s_IBSy/objective/Trial#00068/


# EOF
