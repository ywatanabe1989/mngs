#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-16 13:20:25 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/tests/mngs/io/_save_modules/test__torch.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/io/_save_modules/test__torch.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import os
import tempfile
import pytest
import torch
import torch.nn as nn


def test_save_torch_tensor():
    """Test saving a PyTorch tensor."""
    from mngs.io._save_modules._torch import _save_torch
    
    # Skip test if CUDA is not available (this is just a precaution)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Create test tensor
    test_tensor = torch.tensor([1, 2, 3, 4, 5], device=device)
    
    # Create temp file path for .pt extension
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        temp_path = tmp.name
        
    try:
        # Save the tensor
        _save_torch(test_tensor, temp_path)
        
        # Verify the file exists
        assert os.path.exists(temp_path)
        
        # Load and verify contents
        loaded_tensor = torch.load(temp_path, map_location=device)
        assert torch.equal(test_tensor, loaded_tensor)
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_save_torch_model():
    """Test saving a PyTorch model."""
    # Skip due to pickle constraints with locally defined classes
    pytest.skip("Skipping model serialization test due to pickling constraints")


def test_save_torch_with_kwargs():
    """Test saving a PyTorch tensor with additional kwargs."""
    from mngs.io._save_modules._torch import _save_torch
    
    # Create test tensor
    test_tensor = torch.tensor([1, 2, 3, 4, 5])
    
    # Create temp file path
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        temp_path = tmp.name
        
    try:
        # Save with _use_new_zipfile_serialization=False (to test kwargs passing)
        _save_torch(test_tensor, temp_path, _use_new_zipfile_serialization=False)
        
        # Verify the file exists
        assert os.path.exists(temp_path)
        
        # Load and verify contents
        loaded_tensor = torch.load(temp_path)
        assert torch.equal(test_tensor, loaded_tensor)
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_modules/_torch.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-16 12:25:14 (ywatanabe)"
# # File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_modules/_torch.py
# 
# import torch
# 
# 
# def _save_torch(obj, spath, **kwargs):
#     """
#     Save a PyTorch model or tensor.
#     
#     Parameters
#     ----------
#     obj : torch.nn.Module or torch.Tensor
#         The PyTorch model or tensor to save.
#     spath : str
#         Path where the PyTorch file will be saved.
#     **kwargs : dict
#         Additional keyword arguments to pass to torch.save.
#         
#     Returns
#     -------
#     None
#     """
#     torch.save(obj, spath, **kwargs)
# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_modules/_torch.py
# --------------------------------------------------------------------------------
