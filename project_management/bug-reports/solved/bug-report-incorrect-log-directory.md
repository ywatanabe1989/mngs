# Bug Report: Incorrect Log Directory Location

**Date**: 2025-05-30
**Reporter**: Claude (AI Assistant)
**Severity**: High
**Status**: Open

## Description

MNGS logs are being saved to the wrong directory. When running a script like `examples/mngs_framework.py`, logs should be saved relative to the script location (e.g., `examples/mngs_framework_out/`), but they are currently being saved to the mngs module directory instead.

## Current Behavior

When running:
```bash
python examples/mngs_framework.py
```

Logs are saved to:
```
/data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/gen/_start_out/FINISHED_SUCCESS/{ID}/
```

## Expected Behavior

Logs should be saved relative to the calling script:
```
./examples/mngs_framework_out/FINISHED_SUCCESS/{ID}/
```

## Root Cause Analysis

1. **Path Resolution Issue**: The `_start.py` file is using its own location instead of the calling script's location
2. **File Path Handling**: When `file` parameter is provided (e.g., `__FILE__` from mngs_framework.py), it should be used to determine the output directory
3. **Error in _start.py**: Line 290 has undefined variables (`_sdir` and `sfname`)

## Technical Details

### Key Code Section (src/mngs/gen/_start.py)
```python
# Lines 271-287
if sdir is None:
    # Define __file__
    if file:
        THIS_FILE = file
    else:
        THIS_FILE = inspect.stack()[1].filename
        if "ipython" in THIS_FILE:
            THIS_FILE = f"/tmp/{_os.getenv('USER')}.py"

    # Define sdir
    sdir = clean_path(
        _os.path.splitext(THIS_FILE)[0] + f"_out/RUNNING/{ID}/"
    )
```

### Issues Found

1. The code correctly sets `THIS_FILE` from the `file` parameter
2. However, the resulting path may be absolute when it should be relative to the working directory
3. Line 290 has a bug: `_clear_python_log_dir(_sdir + sfname + "/")` - undefined variables

## Reproduction Steps

1. Navigate to mngs repository
2. Run: `python examples/mngs_framework.py`
3. Observe that logs are saved to the wrong directory

## Proposed Fix

1. Ensure `THIS_FILE` is converted to a path relative to the current working directory
2. Fix line 290 to use `sdir` instead of undefined variables
3. Test that logs are saved next to the calling script

## Impact

- Users cannot find their logs in the expected location
- Logs from different scripts may overwrite each other
- Violates the principle of keeping output near the source script

## Priority

High - This affects the core functionality of the logging system and user experience.