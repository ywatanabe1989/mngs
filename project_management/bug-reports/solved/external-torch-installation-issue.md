# External Bug Report: PyTorch Installation Issue in gPAC Project

## Issue Description
The gPAC project is experiencing PyTorch import failures that prevent mngs from loading properly.

## Error Message
```
ModuleNotFoundError: No module named 'torch._utils_internal'
```

## Root Cause
This error typically indicates a corrupted or incomplete PyTorch installation in the virtual environment.

## Recommended Solutions

### 1. Reinstall PyTorch (Recommended)
```bash
cd /data/gpfs/projects/punim2354/ywatanabe/.claude-worktree/gPAC
source .env-3.10/bin/activate

# Uninstall existing PyTorch
pip uninstall torch torchvision torchaudio -y

# Clear pip cache
pip cache purge

# Reinstall PyTorch (adjust for your CUDA version if needed)
# For CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Create Fresh Virtual Environment (If above doesn't work)
```bash
cd /data/gpfs/projects/punim2354/ywatanabe/.claude-worktree/gPAC

# Backup requirements
pip freeze > requirements_backup.txt

# Remove old environment
rm -rf .env-3.10

# Create new environment
python3.10 -m venv .env-3.10
source .env-3.10/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install mngs
pip install -e /data/gpfs/projects/punim2354/ywatanabe/.claude-worktree/mngs_repo

# Install other requirements
pip install -r requirements.txt  # or requirements_backup.txt
```

### 3. Verify Installation
After reinstalling, verify PyTorch works:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

### 4. Check for Conflicting Installations
Sometimes multiple PyTorch installations can conflict:
```bash
# Check what's installed
pip list | grep torch

# Check for system-wide installations
python -c "import sys; print('\n'.join(sys.path))"
```

## Prevention
To prevent this issue in the future:
1. Always install PyTorch from the official index URLs
2. Pin PyTorch version in requirements.txt
3. Use virtual environments to isolate dependencies
4. Regularly test imports after updates

## Related to mngs
While this is not an mngs bug, mngs imports torch in its types module. Consider making torch an optional dependency or lazy-loading it to avoid blocking mngs usage when PyTorch has issues.