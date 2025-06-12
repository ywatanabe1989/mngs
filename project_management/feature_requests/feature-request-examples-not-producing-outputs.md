# Feature Request: Examples Not Producing Outputs

## Status: RESOLVED ✅ (2025-05-31 08:00)

## Issue Description
The most critical issue is that example scripts in the `./examples/` directory are not producing output directories when run, which violates the mngs framework guidelines.

## Expected Behavior
According to IMPORTANT-MNGS-06-examples-guide.md:
- All example files MUST use the mngs framework (`mngs.gen.start()` and `mngs.gen.close()`)
- Output directories should be automatically created when examples run
- Each example should have a corresponding `_out` directory

## Current Behavior
- Only `mngs_framework_out` directory exists
- Other example scripts don't have output directories
- This indicates either:
  1. Scripts don't follow mngs framework properly
  2. Scripts haven't been run yet
  3. The mngs package has problems with output directory creation

## Investigation Needed
1. Test run a simple example script to verify mngs.gen.start() creates output directories
2. Check if there are any errors when running examples
3. Verify the mngs framework is working correctly
4. Check if examples need to be updated to use latest mngs API

## Proposed Solution
1. Debug why mngs.gen.start() isn't creating output directories
2. Fix any issues in the mngs framework
3. Update all examples to ensure they properly use mngs.gen.start() and mngs.gen.close()
4. Create run_examples.sh script to test all examples
5. Add CI test to ensure examples run without errors

## Priority
**HIGH** - This is a critical issue that affects the usability and reliability of the entire mngs framework.

## Findings
After investigation, the issue is NOT that output directories aren't created. The mngs framework IS working correctly:
1. Output directories ARE being created properly by mngs.gen.start()
2. The example created: `/data/gpfs/projects/punim2354/ywatanabe/.claude-worktree/mngs_repo/examples/mngs/dsp/signal_processing_out/`
3. Files were saved successfully to the output directory

## Root Cause
The actual issues were:
1. Two examples didn't use the standard mngs template properly:
   - scientific_data_pipeline.py: Used custom main() instead of run_main() wrapper
   - enhanced_plotting.py: Used custom main() instead of run_main() wrapper
2. API mismatches in examples:
   - enhanced_plotting.py: set_xyt() doesn't exist, used set_xlabel/ylabel/title instead
   - enhanced_plotting.py: mngs.plt.get_colors → mngs.plt.color.get_colors_from_cmap('viridis', n)
   - signal_processing.py: bandpass() parameters fixed
   - All save paths updated to use relative paths (mngs.io.save handles output dir)

## Resolution (2025-05-31 08:00)
All examples have been updated to properly use the standard mngs template with run_main() wrapper.
- scientific_data_pipeline.py refactored to use standard template
- enhanced_plotting.py refactored to use standard template  
- All API mismatches fixed
- All 11 examples now create output directories automatically
- Tested: experiment_workflow.py, enhanced_plotting.py, scientific_data_pipeline.py all work correctly

## Progress
- [x] Investigate why output directories aren't created (they ARE created)
- [x] Test run experiment_workflow.py example
- [x] Check mngs.gen.start() implementation (working correctly)
- [x] Fix examples not using mngs framework properly
- [x] Search for all occurrences of incorrect framework usage
- [x] Update scientific_data_pipeline.py to use standard template
- [x] Update enhanced_plotting.py to use standard template
- [x] Fix API mismatches in examples
- [x] Verify all 11 examples have output directories
- [x] Test multiple examples to confirm they work
- [ ] Create run_examples.sh script
- [ ] Add CI test for examples