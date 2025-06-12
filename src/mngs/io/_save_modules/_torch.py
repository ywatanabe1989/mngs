#!/usr/bin/env python3
# -*- coding: utf-8 -*-
<<<<<<< HEAD
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
=======
# Timestamp: "2025-05-16 12:25:14 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_modules/_torch.py
>>>>>>> origin/main

import torch


<<<<<<< HEAD
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
=======
def _save_torch(obj, spath, **kwargs):
    """
    Save a PyTorch model or tensor.
    
    Parameters
    ----------
    obj : torch.nn.Module or torch.Tensor
        The PyTorch model or tensor to save.
    spath : str
        Path where the PyTorch file will be saved.
    **kwargs : dict
        Additional keyword arguments to pass to torch.save.
        
    Returns
    -------
    None
    """
    torch.save(obj, spath, **kwargs)
>>>>>>> origin/main
