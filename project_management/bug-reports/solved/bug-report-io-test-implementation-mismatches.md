# Bug Report: IO Test Implementation Mismatches

## Issue Description
The comprehensive test suite for mngs.io module contains multiple tests that expect functionality that doesn't exist in the actual implementation, causing 12 out of 22 tests to fail.

## Reproduction Steps
1. Run: `python -m pytest tests/mngs/io/test__io_comprehensive.py -v`
2. Observe 12 test failures

## Root Cause Analysis

### 1. NPZ File Loading (FIXED)
- **Issue**: `_load_npz` returned a list of values instead of preserving dictionary structure
- **Impact**: `test_numpy_roundtrip` failed
- **Fix Applied**: Changed to return NpzFile object preserving key access

### 2. Text File Loading
- **Issue**: `_load_txt` returns list of stripped lines, not original text
- **Test Expects**: Original text with newlines preserved
- **Current Behavior**: `['Line 1', 'Line 2', 'Line 3']`
- **Expected Behavior**: `'Line 1\nLine 2\nLine 3\n'`

### 3. Excel Format Support
- **Issue**: `.xlsx` format is NOT supported by save function
- **Test Expects**: Full Excel save/load support
- **Error**: "Unsupported file format. /tmp/.../data.xlsx was not saved."

### 4. HDF5 Format Requirements
- **Issue**: HDF5 save expects dict input only
- **Test May Pass**: Other data types
- **Current Implementation**: Only handles dict with specific structure

### 5. Missing Features in Tests
- Glob pattern loading
- Compressed file handling (.gz support)
- Special encoding support
- Empty data handling for various formats

## Impact
- Test coverage appears artificially low due to mismatched expectations
- Developers may implement features that already exist differently
- Confusion about actual capabilities of the IO module

## Proposed Solutions

### Option 1: Update Tests to Match Implementation
- Modify text tests to expect list of lines
- Remove Excel tests or mark as expected failures
- Adjust other tests to match actual behavior

### Option 2: Implement Missing Features
- Add Excel support using openpyxl or xlsxwriter
- Modify text loader to preserve original format
- Enhance compression and encoding support

### Option 3: Hybrid Approach
- Fix critical issues (like NPZ - already done)
- Update tests for current behavior
- Create feature requests for missing functionality

## Priority
High - This blocks accurate assessment of test coverage and may mislead developers about module capabilities.

## Related Files
- `/src/mngs/io/_save.py` - Main save implementation
- `/src/mngs/io/_load.py` - Main load implementation
- `/src/mngs/io/_load_modules/` - Format-specific loaders
- `/tests/mngs/io/test__io_comprehensive.py` - Comprehensive test suite

## Recommendation
Start with Option 3 (Hybrid Approach):
1. Keep the NPZ fix (already applied)
2. Update text file tests to match current behavior
3. Skip or mark Excel tests as expected failures
4. Create feature requests for Excel support and other missing features
5. Document actual behavior in API docs