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

- src/mngs/str/_gen_timestamp.py:26 - '\bfilename\b' should be 'filepath'
- src/mngs/str/_gen_timestamp.py:27 - '\bfilename\b' should be 'filepath'
- src/mngs/str/__init__.py:14 - '\bfilename\b' should be 'filepath'
- src/mngs/str/__init__.py:15 - '\bfilename\b' should be 'filepath'
- src/mngs/str/__init__.py:16 - '\bfilename\b' should be 'filepath'
- src/mngs/str/__init__.py:26 - '\bfilename\b' should be 'filepath'
- src/mngs/plt/ax/_style/_set_ticks.py:269 - '\bfs\b' should be 'sample_rate'
- src/mngs/plt/ax/_style/_set_ticks.py:270 - '\bfs\b' should be 'sample_rate'
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

## Missing Docstrings
Found functions/classes without docstrings (showing first 20):

- src/mngs/str/_mask_api_key.py:7 - mask_api
- src/mngs/plt/ax/_style/_sci_note.py:59 - __init__
- src/mngs/plt/ax/_style/_share_axes.py:208 - set_xlims
- src/mngs/plt/ax/_style/_share_axes.py:231 - set_ylims
- src/mngs/plt/ax/_style/_share_axes.py:255 - main
- src/mngs/plt/ax/_plot/_plot_rectangle.py:16 - plot_rectangle
- src/mngs/plt/ax/_plot/_plot_joyplot.py:20 - plot_joyplot
- src/mngs/plt/ax/_plot/_plot_violin.py:297 - half_violin
- src/mngs/plt/utils/_close.py:18 - close
- src/mngs/plt/_subplots/_SubplotsWrapper.py:29 - __init__
- src/mngs/plt/_subplots/_SubplotsWrapper.py:34 - __call__
- src/mngs/plt/_subplots/_AxisWrapper.py:22 - AxisWrapper
- src/mngs/plt/_subplots/_AxisWrapper.py:25 - __init__
- src/mngs/plt/_subplots/_AxisWrapper.py:45 - get_figure
- src/mngs/plt/_subplots/_AxisWrapper.py:48 - __getattr__
- src/mngs/plt/_subplots/_AxisWrapper.py:135 - __dir__
- src/mngs/plt/_subplots/_AxisWrapper.py:62 - wrapper
- src/mngs/plt/_subplots/_AxisWrapper.py:116 - fallback_method
- src/mngs/plt/_subplots/_export_as_csv.py:23 - export_as_csv
- src/mngs/plt/_subplots/_export_as_csv.py:49 - format_record

Note: Private functions (_name) are excluded from this check.

## Summary
- Total naming issues: 55
- File naming issues: 1
- Function naming issues: 14
- Class naming issues: 0
- Abbreviation issues: 20+
- Missing docstrings: 20+