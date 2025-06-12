#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-12 13:45:00 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/.claude-worktree/mngs_repo/src/mngs/io/_save_modules/_torch.py
# ----------------------------------------
import os

__FILE__ = "./src/mngs/io/_save_modules/_torch.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
PyTorch saving functionality for mngs.io.save
"""

import torch


def save_torch(obj, spath, **kwargs):
    """Handle PyTorch model/tensor file saving.
    
    Parameters
    ----------
    obj : torch.Tensor, torch.nn.Module, dict, or Any
        Object to save as PyTorch file. Can be a tensor, model, state_dict, or any serializable object.
    spath : str
        Path where PyTorch file will be saved
    **kwargs
        Additional keyword arguments passed to torch.save()
    """
    torch.save(obj, spath, **kwargs)


# EOF