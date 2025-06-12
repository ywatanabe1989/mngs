# MNGS Naming Inconsistencies Analysis

## File Naming Issues
Found 1 file naming issues:

- Not snake_case: src/mngs/ai/optim/Ranger_Deep_Learning_Optimizer/ranger/ranger913A.py

## Function Naming Issues
Found 14 functions not following snake_case:

- src/mngs/gen/_close.py:59 - _escape_ANSI_from_log_files
- src/mngs/gists/_SigMacro_toBlue.py:1 - SigMacro_toBlue
- src/mngs/gists/_SigMacro_processFigure_S.py:1 - SigMacro_processFigure_S
- src/mngs/dsp/utils/pac.py:57 - plot_PAC_mngs_vs_tensorpac
- src/mngs/types/_is_listed_X.py:13 - is_listed_X
- src/mngs/ai/classification_reporter.py:157 - calc_bACC
- src/mngs/ai/classification_reporter.py:229 - calc_AUCs
- src/mngs/ai/classification_reporter.py:267 - _calc_AUCs_binary
- src/mngs/ai/plt/_conf_mat.py:199 - calc_bACC_from_cm
- src/mngs/ai/metrics/_bACC.py:12 - bACC
... and 4 more

## Class Naming Issues
✅ All classes follow PascalCase!

## Inconsistent Abbreviations
Found inconsistent abbreviations (showing first 20):

- src/mngs/str/__init__.py:14 - '\bfilename\b' should be 'filepath'
- src/mngs/str/__init__.py:15 - '\bfilename\b' should be 'filepath'
- src/mngs/str/__init__.py:16 - '\bfilename\b' should be 'filepath'
- src/mngs/str/__init__.py:26 - '\bfilename\b' should be 'filepath'
- src/mngs/plt/ax/_style/_set_ticks.py:233 - '\bfs\b' should be 'sample_rate'
- src/mngs/plt/ax/_style/_set_ticks.py:234 - '\bfs\b' should be 'sample_rate'
- src/mngs/plt/utils/_calc_nice_ticks.py:23 - '\bnum_' should be 'n_'
- src/mngs/plt/utils/_calc_nice_ticks.py:40 - '\bnum_' should be 'n_'
- src/mngs/plt/utils/_calc_nice_ticks.py:72 - '\bnum_' should be 'n_'
- src/mngs/plt/utils/_calc_nice_ticks.py:84 - '\bnum_' should be 'n_'
- src/mngs/plt/utils/_calc_nice_ticks.py:85 - '\bnum_' should be 'n_'
- src/mngs/plt/_subplots/__init__.py:20 - '\bfilename\b' should be 'filepath'
- src/mngs/plt/_subplots/__init__.py:21 - '\bfilename\b' should be 'filepath'
- src/mngs/plt/_subplots/__init__.py:22 - '\bfilename\b' should be 'filepath'
- src/mngs/plt/_subplots/__init__.py:37 - '\bfilename\b' should be 'filepath'
- src/mngs/plt/_subplots/_FigWrapper.py:55 - '\bfname\b' should be 'filepath'
- src/mngs/plt/_subplots/_FigWrapper.py:58 - '\bfname\b' should be 'filepath'
- src/mngs/plt/_subplots/_FigWrapper.py:62 - '\bfname\b' should be 'filepath'
- src/mngs/plt/color/_interpolate.py:18 - '\bnum_' should be 'n_'
- src/mngs/plt/color/_interpolate.py:22 - '\bnum_' should be 'n_'

## Missing Docstrings
Found functions/classes without docstrings (showing first 20):

- src/mngs/__main__.py:46 - main
- src/mngs/str/_gen_timestamp.py:9 - gen_timestamp
- src/mngs/str/_mask_api.py:7 - mask_api
- src/mngs/str/_mask_api_key.py:7 - mask_api
- src/mngs/str/_gen_ID.py:11 - gen_id
- src/mngs/str/_search.py:61 - to_list
- src/mngs/str/_print_debug.py:11 - print_debug
- src/mngs/plt/ax/_style/_sci_note.py:17 - OOMFormatter
- src/mngs/plt/ax/_style/_sci_note.py:18 - __init__
- src/mngs/plt/ax/_style/_add_marginal_ax.py:17 - add_marginal_ax
- src/mngs/plt/ax/_style/_force_aspect.py:16 - force_aspect
- src/mngs/plt/ax/_style/_set_ticks.py:21 - set_ticks
- src/mngs/plt/ax/_style/_map_ticks.py:92 - numeric_example
- src/mngs/plt/ax/_style/_map_ticks.py:113 - string_example
- src/mngs/plt/ax/_style/_add_panel.py:51 - panel
- src/mngs/plt/ax/_style/_share_axes.py:18 - sharexy
- src/mngs/plt/ax/_style/_share_axes.py:23 - sharex
- src/mngs/plt/ax/_style/_share_axes.py:28 - sharey
- src/mngs/plt/ax/_style/_share_axes.py:33 - get_global_xlim
- src/mngs/plt/ax/_style/_share_axes.py:64 - get_global_ylim

Note: Private functions (_name) are excluded from this check.

## Summary
- Total naming issues: 55
- File naming issues: 1
- Function naming issues: 14
- Class naming issues: 0
- Abbreviation issues: 20+
- Missing docstrings: 20+