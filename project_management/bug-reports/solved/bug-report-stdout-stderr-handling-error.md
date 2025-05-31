# Bug Report: stdout/stderr Handling Error in mngs_framework.py [SOLVED]

## Issue Description
When running `./examples/mngs_framework.py`, the script shows error messages related to stdout/stderr handling:
- `Exception ignored in: Exception ignored in sys.unraisablehook`
- This occurs during the cleanup phase when the script exits

## Reproduction Steps
1. Navigate to the mngs_repo directory
2. Run: `python examples/mngs_framework.py`
3. Observe the error message at the beginning of the output

## Expected Behavior
The script should run cleanly without any exception warnings related to stdout/stderr handling.

## Actual Behavior
The script outputs:
```
Exception ignored in: Exception ignored in sys.unraisablehook
```
followed by environment variable output and then the expected script output.

## Root Cause Analysis
The issue appears to be in the `Tee` class in `/src/mngs/gen/_tee.py`:

1. The `Tee` class implements `__del__` method which tries to close the log file
2. During Python shutdown, the cleanup order is not guaranteed, and some objects may already be destroyed
3. When `__del__` is called during shutdown, it may fail to properly close resources
4. The error is being suppressed by Python's unraisable exception hook

## Proposed Solution
1. Implement proper context manager protocol (`__enter__` and `__exit__`) for the Tee class
2. Ensure proper cleanup in `mngs.gen.close()` function
3. Add explicit cleanup before Python shutdown
4. Handle exceptions more gracefully in the `__del__` method

## Impact
- Low severity - the script still runs correctly
- High visibility - the error appears on every run
- Affects user experience and confidence in the framework

## Priority
Medium - While functionality is not affected, the error message creates a poor user experience and may mask other important messages.

## Related Files
- `/src/mngs/gen/_tee.py` - Contains the Tee class with problematic cleanup
- `/src/mngs/gen/_start.py` - Initializes the Tee objects
- `/src/mngs/gen/_close.py` - Should handle proper cleanup
- `/examples/mngs_framework.py` - Example script showing the issue

## Solution Implemented
Fixed the issue by:

1. **Improved `__del__` method in Tee class** (`/src/mngs/gen/_tee.py`):
   - Added checks to prevent cleanup during Python shutdown
   - Check if file is already closed before attempting to close
   - Set `_log_file = None` after closing to prevent double-close

2. **Enhanced cleanup in close function** (`/src/mngs/gen/_close.py`):
   - Added explicit flush before closing
   - Better error handling during cleanup

## Testing
- Confirmed no more "Exception ignored" errors
- Verified stdout/stderr are still properly logged
- Tested multiple runs to ensure stability

## Status
**SOLVED** - 2025-05-30