# Bug Report: Fixed Tee Logging Path Issue

**Date**: 2025-05-30
**Reporter**: Claude (AI Assistant)  
**Severity**: High
**Status**: FIXED

## Issue Description

The `mngs.gen._tee.py` module was using `__file__` (which refers to the `_tee.py` file itself) instead of the calling script's location when determining where to save log files. This caused logs to always be saved relative to the mngs module location instead of the script being executed.

## Root Cause

In `src/mngs/gen/_tee.py`, lines 185-187:
```python
# Before (incorrect):
if "ipython" in __file__:  # Wrong: __file__ refers to _tee.py
    THIS_FILE = f"/tmp/{_os.getenv('USER')}.py"
sdir = clean_path(_os.path.splitext(__file__)[0] + "_out")  # Wrong: uses _tee.py location
```

## Fix Applied

Changed to use `THIS_FILE` variable which correctly captures the calling script's location:
```python
# After (correct):
if "ipython" in THIS_FILE:  # Correct: THIS_FILE is from inspect.stack()[1].filename
    THIS_FILE = f"/tmp/{_os.getenv('USER')}.py"
sdir = clean_path(_os.path.splitext(THIS_FILE)[0] + "_out")  # Correct: uses calling script location
```

Additionally, added explicit flush calls in the `write` method to ensure logs are written immediately:
```python
self._log_file.write(data)
self._log_file.flush()  # Ensure immediate write
```

## Test Results

The fix was tested and confirmed working:
- Logs are now correctly saved relative to the calling script
- For a script at `/path/to/script.py`, logs are saved in `/path/to/script_out/logs/`
- The flush ensures logs are written immediately, not buffered

## Impact

This fix ensures that:
1. Logs are saved in the expected location next to the calling script
2. Different scripts maintain separate log directories
3. Users can easily find their logs
4. The behavior matches the documented expectations

## Files Modified

- `src/mngs/gen/_tee.py`: Fixed variable references and added flush calls

## Status

FIXED - The logging path issue has been resolved.