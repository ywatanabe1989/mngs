# Quick Migration Guide: MNGS → SciTeX

## For Your Personal Use - Simple Steps

### Option 1: Full Rebranding (Recommended)
```bash
# Run the automated script
./rebrand_to_scitex.sh

# This will:
# - Backup your current state
# - Rename all occurrences of mngs to scitex
# - Update imports, documentation, and config files
# - Test that the new import works
```

### Option 2: Update Existing Projects Only
```bash
# For a specific project directory
python update_mngs_imports.py /path/to/your/project

# For current directory
python update_mngs_imports.py
```

### Option 3: Temporary Compatibility (While Transitioning)
Add to your `~/.bashrc` or `~/.zshrc`:
```bash
export PYTHONPATH="$HOME/.mngs_compat:$PYTHONPATH"
```

Then create `~/.mngs_compat/mngs.py`:
```python
# Redirect imports
import scitex
import sys
sys.modules['mngs'] = scitex
```

### Quick Import Changes

| Old | New |
|-----|-----|
| `import mngs` | `import scitex` |
| `import mngs as mg` | `import scitex as stx` |
| `from mngs.io import save` | `from scitex.io import save` |
| `mngs.plt.subplots()` | `scitex.plt.subplots()` |

### Testing After Migration
```bash
# Quick test
python -c "import scitex; print(scitex.__version__)"

# Run your test suite
pytest tests/

# Test a specific module
python -c "from scitex.io import save, load; print('✓ IO works')"
python -c "from scitex.plt import subplots; print('✓ Plotting works')"
```

### If Something Breaks
1. The original code is tagged: `git checkout v1.11.0-final-mngs`
2. Use the compatibility import: `python mngs_compatibility.py`
3. Gradually update imports file by file

### PyPI Publishing (When Ready)
```bash
# Update version in pyproject.toml
# Build
python -m build

# Upload to test PyPI first
python -m twine upload --repository testpypi dist/*

# Test installation
pip install -i https://test.pypi.org/simple/ scitex

# Upload to PyPI
python -m twine upload dist/*
```

### Your Most Common Use Cases

1. **For notebooks**: Add at the top
   ```python
   try:
       import scitex as stx
   except ImportError:
       import mngs as stx  # Fallback
   ```

2. **For scripts**: Use the update script
   ```bash
   python update_mngs_imports.py your_script.py
   ```

3. **For new projects**: Just use scitex from the start
   ```python
   import scitex as stx
   ```

---
Remember: Since you're the main user, you can take a gradual approach and update projects as needed!