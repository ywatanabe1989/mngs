#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-02 23:32:31 (ywatanabe)"
# /home/ywatanabe/proj/_mngs_repo_openhands/src/mngs/tex/_to_vec.py

def to_vec(v_str):
    r"""
    Convert a string to LaTeX vector notation.

    Example
    -------
    vector = to_vec("AB")
    print(vector)  # Outputs: \overrightarrow{\mathrm{AB}}

    Parameters
    ----------
    vector_string : str
        String representation of the vector

    Returns
    -------
    str
        LaTeX representation of the vector
    """
    return f"\\overrightarrow{{\\mathrm{{{v_str}}}}}"
