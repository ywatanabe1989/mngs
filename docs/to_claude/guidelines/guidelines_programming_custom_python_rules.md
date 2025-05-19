<!-- ---
!-- Timestamp: 2025-05-17 07:13:59
!-- Author: ywatanabe
!-- File: /ssh:ywatanabe@sp:/home/ywatanabe/.dotfiles/.claude/to_claude/guidelines/guidelines_programming_custom_python_rules.md
!-- --- -->


# Python-specific Rules
================================================================================

## Python MNGS Package
### Overview
- Custom Python utility package for standardizing scientific analyses
- Features predefined directory structure and conventions
- Manages configuration, environment, and file organization
- Include versatile functions and reuse among projects
- Located in `~/proj/mngs_repo/src/mngs` and `git@github.com:ywatanabe1989:mngs`
- Installed via pip development mode (`pip install -e ~/proj/mngs_repo`)
- `mngs` means monogusa, meaning lazy in Japanese.

### Key MNGS Functions
- **`mngs.gen.start(...)`** - Initialize environment: e.g., logging, paths, matplotlib, random seed
- **`mngs.gen.close(...)`** - Clean up environment
- **`mngs.io.load_configs()`** - Load YAML files under ./config as a dot-accessible dictionary
  - Centralize variables here (paths in PATH.yaml, project-specific params in project.yaml)
- **`mngs.io.load()`** - Load data using relative path from project root
  - Ensure to load from (symlinks of) `./data` directory, not from `./scripts`
  - Use CONFIG.PATH.... to specify paths

- !!! IMPORTANT !!! **`mngs.io.save(obj, ./relative/spath.ext, symlink_from_cwd=True)`** - Save with automatic format handling
  - e.g., `mngs.io.save(obj, ./relative/spath.ext, symlink_from_cwd=True)` in `/path/to/script.py` creates:
    - `/path/to/script_out/relative/spath/from/the/script.ext`
    - `REPOSITORY_ROOT/relative/spath/from/the/script.ext` -> `/path/to/script_out/relative/spath/from/the/script.ext`
    - This do not need to handle `mkdir`; it is automatically created by `mngs.io.save`
    - Start from "./" for the relative path; it will save in correct places
  - `symlink_from_cwd=True` creates symlink from current working directory (, which should be`REPO_ROOT`)

  - Organize data (and symlinks) under the ./data directory
- **`mngs.plt.subplots(ncols=...)`** - Matplotlib wrapper that tracks plotting data
  - Plotted data will be saved in CSV with the same name as the figure
  - CSV file will be the same format expected in plotting software SigmaPlot
- **`mngs.str.printc(message, c='blue')`** - Colored console output
  - Acceptable colors: 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white', 'grey'
  - Default is None, which means no color
- **`mngs.stats.p2stars(p_value)`** - Convert p-values to significance stars
- **`mngs.stats.fdr_correction(results)`** - Apply FDR correction for multiple comparisons
- **`mngs.pd.round(df, factor=3)`** - Round numeric values in dataframe with specified precision

#### `mngs.io.load()`
- Ensure to use relative path from the project root
- Ensure to load data from (symlinks of) `./data` directory, instead of from `./scripts`
- Use CONFIG.PATH.... to load data

#### `mngs.io.save(obj, spath, symlink_from_cwd=True)`
- From the extension of spath, saving code will be handled automatically
- Note that save path must be relative to use symlink_from_cwd=True
- Especially, this combination is expected because this organize data (and symlinks) under the ./data directory: `mngs.io.save(obj, ./data/path/to/data.ext, symlink_from_cwd=True)`
 
#### `mngs.plt.subplots(ncols=...)`
  - Use CONFIG.PATH.... to save data
  - This is a matplotlib.plt wrapper. 
  - Plotted data will be saved in the csv with the same name of the figure
  - CSV file will be the same format expected in a plotting software, SigmaPlot
#### `mngs.str.printc(message, c='blue)`
  - Acceptable colors: 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white', or 'grey' 
    - (default is None, which means no color)

## Python MNGS Rules
- We will call the following format as `mngs format`
  - Ensure to use the `run_main()` function with main guard
  - ANY MODIFICATION, EVEN ONE LETTER, IS PERMITTED FOR THIS RUN_MAIN FUNCTION
    ```python
    def run_main() -> None:
        """Initialize mngs framework, run main function, and cleanup."""
        global CONFIG, CC, sys, plt

        import sys
        import matplotlib.pyplot as plt
        import mngs

        args = parse_args()

        # Start mngs framework
        CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
            sys,
            plt,
            args=args,
            file=__file__,
            sdir_suffix=None,
            verbose=False,
            agg=True,
        )

        # Main
        exit_status = main(args)

        # Close the mngs framework
        mngs.gen.close(
            CONFIG,
            verbose=False,
            notify=False,
            message="",
            exit_status=exit_status,
        )

    if __name__ == '__main__':
        run_main()
    ```

    - This is quite important that in the mngs project, this handles:
      - stdout/stderr direction, logging, configuration saving, arguments saving, and so on
- Implement functions and classes under the """Functions & Classes""" tag
- Combine functions and classes in the main function and call it from run_main() function

!!! IMPORTANT !!!
ENSURE TO USE THIS MNGS FORMAT FOR ANY PYTHON SCRIPT FILE.
- ENSURE TO IMPLEMENT `main()` and WRAP IT WITH THE `run_main()` FUNCTION
- THE `run_main()` FUNCTION MUST BE IDENTICAL WITHOUT ANY LETTER MODIFIED
  - This is quite important that in the mngs project, this handles:
    - configuration
    - seeds fixation
    - stdout/stderr handling
    - logging
    - saving directory handling
    - symbolic link handling
    - arguments saving
    - and more
- Combine functions and classes in the main function and call it from run_main() function


## Python MNGS Project Structure for Scientific Research
- DO NOT CREATE ANY DIRECTORIES IN PROJECT ROOT
- However, creating child directories under predefined directories is recommended
- Hide unnecessary files and directories
- Organize project structures by moving, hiding, combining files
  - When path changed, ensure to update `./config/PATH.yaml` correspondingly
- Produced data from scripts should be placed near the source scripts
  - This can be easily handled by `mngs.io.save`
  - Symbolic links should be organized in the data directory, referencing to the original file under the scripts directory

My Python utility package, `mngs`, is desined to standardize scientific analyses and applications.
mngs project features predefined directory structure - `./config`, `./data`, `./scripts`

Config:
    - Centralized YAML configuration files (e.g., `./config/PATH.yaml`)
Data:
    - Centralized data files (e.g., `./data/mnist`)

Scripts:
    - Script files (.py and .sh)
    - Directly linked outputs (artifacts and logs)
    - For example, 
        - `./scripts/mnist/plot_images.py` produces:
            - `./scripts/mnist/plot_images_out/data/figures/mnist_digits.jpg`
            - `./scripts/mnist/plot_images_out/RUNNING`
            - `./scripts/mnist/plot_images_out/FINISHED_SUCCESS`
            - `./scripts/mnist/plot_images_out/FINISHED_FAILED`
    - Symlink to `./data` directory can be handled by `mngs.io.save(obj, spath, symlink_from_cwd=True)`
    - Ensure current working directory (`cwd`) is always the project root

## Python MNGS Project Structure Format
```plaintext
<project root>
|-- .git
|-- .gitignore
|-- .env
|   `-- bin
|       |-- activate
|       `-- python
|-- config
|   |-- <file_name>.yaml
|   |...
|   `-- <file_name>.yaml
|-- data
|   `-- <dir_name>
|        |-- <file_name>.npy -> ../../scripts/<script_name>_out/<file_name>.npy (handled by mngs.io.save)
|        |...
|        `-- <file_name>.mat -> ../../scripts/<script_name>_out/<file_name>.mat (handled by mngs.io.save)
|-- mgmt
|   |-- original_plan.md (DO NOT MODIFY)
|   |-- progress.mmd (Mermaid file for visualization; update this as the project proceed)
|   `-- progress.md (Main progree management file in markdown; update this as the project proceed)
|-- README.md
|-- requirements.txt (for the .env; update this as needed)
`-- scripts
    `-- <dir-name>
        |-- <script_name>.py
        |-- <script_name>_out
        |    |-- <file_name>.npy
        |    |-- <file_name>.mat
        |    |-- FINISHED_SUCCESS (handled by mngs.io.save)
        |    |    |-- YYYY-MM-DD-hhmmss_<call-id>/(stdout, stderr, and configs)
        |    |    `-- YYYY-MM-DD-hhmmss_<call-id>/(stdout, stderr, and configs)
        |    |-- FINISHED_FAILURE (handled by mngs.io.save)
        |    |    `-- YYYY-MM-DD-hhmmss_<call-id>/(stdout, stderr, and configs)
        |    `-- RUNNING (handled by mngs.io.save)
        |         `-- YYYY-MM-DD-hhmmss_<call-id>/(stdout, stderr, and configs)
        |-- <script_name>.sh 
        `-- .<script_name>.sh.log 
```

## Python MNGS Config Format and Usage

### Python YAML Configuration Example
```yaml
# Time-stamp: "2025-01-18 00:00:34 (ywatanabe)"
# File: ./config/COLORS.yaml

COLORS:
  SEIZURE_TYPE:
    "1": "red"
    "2": "orange"
    "3": "pink"
    "4": "gray"
```

```yaml
# Time-stamp: "2025-01-18 00:00:34 (ywatanabe)"
# File: ./config/PATH.yaml

PATH:
  ECoG:
    f"./data/patient_{patient_id}/{date}/ecog_signal.npy"
```

``` python
import mngs
CONFIG = mngs.io.load_configs()
print(CONFIG.COLORS.SEIZURE_TYPE) # {"1": "red", "2", "orange", "3": "pink", "4": "gray"}
print(CONFIG.PATH.ECoG) # f"./data/{patient_id}/{date}/ecog_signal.npy"
patient_id = "001"
date = "2025_0101"
print(eval(CONFIG.PATH.ECoG)) # "./data/patient_001/2025_0101/ecog_signal.npy"
```

## Python MNGS Script Format
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-03 10:33:13 (ywatanabe)"
# File: placeholder.py

__file__ = "placeholder.py"

"""
Functionalities:
  - Does XYZ
  - Does XYZ
  - Does XYZ
  - Saves XYZ

Dependencies:
  - scripts:
    - /path/to/script1
    - /path/to/script2
  - packages:
    - package1
    - package2
IO:
  - input-files:
    - /path/to/input/file.xxx
    - /path/to/input/file.xxx

  - output-files:
    - /path/to/input/file.xxx
    - /path/to/input/file.xxx
"""

"""Imports"""
import os
import sys
import argparse

"""Warnings"""
# mngs.pd.ignore_SettingWithCopyWarning()
# warnings.simplefilter("ignore", UserWarning)
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore", UserWarning)

"""Parameters"""
# from mngs.io import load_configs
# CONFIG = load_configs()

"""Functions & Classes"""
def main(args):
    pass

import argparse
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    import mngs
    script_mode = mngs.gen.is_script()
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    mngs.str.printc(args, c='yellow')
    return args

def run_main() -> None:
    """Initialize mngs framework, run main function, and cleanup.

    mngs framework manages:
      - Parameters defined in yaml files under `./config dir`
      - Setting saving directory (/path/to/file.py -> /path/to/file.py_out/)
      - Symlink for `./data` directory
      - Logging timestamp, stdout, stderr, and parameters
      - Matplotlib configurations (also, `mngs.plt` will track plotting data)
      - Random seeds

    THUS, DO NOT MODIFY THIS RUN_MAIN FUNCTION
    """
    global CONFIG, CC, sys, plt

    import sys
    import matplotlib.pyplot as plt
    import mngs

    args = parse_args()

    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        args=args,
        file=__file__,
        sdir_suffix=None,
        verbose=False,
        agg=True,
    )

    exit_status = main(args)

    mngs.gen.close(
        CONFIG,
        verbose=False,
        notify=False,
        message="",
        exit_status=exit_status,
    )

if __name__ == '__main__':
    run_main()

# EOF
```


## Python MNGS Statistics Rules
- Report statistical results with p-value, stars, sample size, effect size, test name, statistic, and null hypothesis
- Use `mngs.stats.p2stars` for p-values with stars
- Implement FDR correction for multiple comparisons
- Round statistical values by factor 3 (.3f format)
- Write statistical values in italic font (e.g., \textit{n})
  ``` python
  results = {
       "p_value": pval,
       "stars": mngs.stats.p2stars(pval), # Float to string: e.g., 0.02 -> "*", 0.009 -> "**"
       "n1": n1,
       "n2": n2,
       "dof": dof,
       "effsize": effect_size,
       "test_name": test_name_text,
       "statistic": statistic_value,
       "H0": null_hypothes_text,
  }
  ```
  ```
  >>> mngs.stats.p2stars(0.0005)
  '***'
  >>> mngs.stats.p2stars("0.03")
  '*'
  >>> mngs.stats.p2stars("1e-4")
  '***'
  >>> df = pd.DataFrame({'p_value': [0.001, "0.03", 0.1, "NA"]})
  >>> mngs.stats.p2stars(df)
     p_value
  0    0.001 ***
  1    0.030   *
  2    0.100
  3       NA  NA
  ```
- For multiple comparisons, please use the FDR correction with `mngs.stats.fdr_correction`:
  - # mngs.stats.fdr_correctiondef
  ``` python
  fdr_correction(results: pd.DataFrame) -> pd.DataFrame:
      if "p_value" not in results.columns:
          return results
      _, fdr_corrected_pvals = fdrcorrection(results["p_value"])
      results["p_value_fdr"] = fdr_corrected_pvals
      results["stars_fdr"] = results["fdr_p_value"].apply(mngs.stats.p2stars)
      return results
  ```

## Python Path Rules
- Use relative path from the project root
- Relative paths should start with dots, like `./relative/example.py` or `../relative/example.txt`
- Use relative paths but full paths are acceptable when admirable
- All Scripts are assumed to be executed from the project root (e.g., ./scripts/example.sh)
- Do Not Repeat Yourself
  - Use symbolic links wisely, especially for large data to keep clear organization and easy navigation
  - Prepare `./scripts/utils/<versatile_func.py>` for versatile and reusable code
    - Then, for example, `from scripts.utils.versatile_func import versatile_func` from main files
  - If versatile code is applicable beyond projects, implement in the `mngs` package, my toolbox
    - `mngs` is in `$HOME/proj/mngs_repo/src/mngs/`
      - It is installed in the `./.env` via a development installation (e.g., `pip install -e $HOME/proj/mngs_repo/`)
- For souce code for the mngs package:
  - Import functions using underscore to keep namespace clean (e.g., import numpy as _np).
  - Use relative import to reduce dependency (e.g., from ..io._load import load)

### MNGS Directory Structure
```
<project root>
|-- .git
|-- .gitignore
|-- .env
|   `-- bin
|       |-- activate
|       `-- python
|-- config
|   |-- <file_name>.yaml
|   |-- <file_name>.yaml
|   |...
|   `-- <file_name>.yaml
|-- data
|   `-- <dir_name>
|        |-- <file_name>.npy -> ../../scripts/path/to/script/script_name_out/<file_name>.npy
|        |-- <file_name>.txt -> ../../scripts/path/to/script/script_name_out/<file_name>.txt
|        |-- <file_name>.csv -> ../../scripts/path/to/script/script_name_out/<file_name>.csv
|        |...
|        `-- <file_name>.mat -> ../../scripts/path/to/script/script_name_out/<file_name>.mat
|-- project_management
|   |-- ORIGINAL_PLAN.md
|   |-- progress.md 
|   |-- progress.mmd
|   |-- progress.gif
|   `-- progress.svg
|-- README.md
|-- requirements.txt
`-- scripts
    `-- <OLD-TIMESTAMP>-<title>
        |-- NNN_<script_name>.py
        |-- NNN_<script_name>_out/<file_name>.npy
        |-- NNN_<script_name>_out/<file_name>.txt
        |-- NNN_<script_name>_out/<file_name>.csv
        `-- run_all.sh
```

### Directory Purpose
- **`./.env`**: Python environment for the project (may be symlinked to shared env)
- **`./.git`**: Git directory
- **`./config`**: Centralized YAML configuration files shared in the project
- **`./data`**: Centralized data files, symbolic links to actual files in scripts
- **`./project_management`**: Project management files
- **`./scripts`**: Executable files (.py, .sh), their output files and logs
- **`.gitignore`**: Ignore large files under `./data` and `./scripts`

### Details of `./config`
- Store in `./config` in YAML format (e.g., `./config/PATH.yaml`)
- All constant variables should be centralized under `./config`
- From Python scripts, access to constants under `./config` by:
- `CONFIG` variable is a custom dot-accessible dictionary
- `CONFIG` variable contains all YAML config files under `./config`
``` python
import mngs
CONFIG = mngs.io.load_configs()
```
- f-strings are acceptable in config YAML files
```python
var_with_f_string = eval(CONFIG.var_with_f_string)
```
- Also, f-string can work as expressions for search
  - For example, 
  mngs.path.glob(`f"./data/mat/Patient_{patient_id}/Data_{year}_{month}_{day}/Hour_{hour}/UTC_{hour}_{minute}_00.mat"`)
  will return path list for matching `./data/mat/Patient_*/Data_*_*_*/Hour_*/UTC_*_*_00.mat"`and corresponding values as dictionary:
  ```python
  {"patient_id": "0001", "year": 2025, "month": 1, ...}
  ```

### Details of `./data`
- Centralize data files shared in the project
  - Store files other than scripts, such raw data and large files (.npy, .csv, .jpg, and .pth data)
    - However, the outputs of a script should be located close to the script
      - e.g., ./scripts/example.py_out/output.jpg
    - Then, symlinked to `./data` directory
      - e.g., ./scripts/example.py_out/data/output.jpg -> ./data/output.jpg
  - This is useful to link scripts and outputs, while keeping centralized data directory structure
  - Large files under `./scripts` or `./data` are git-ignored

### Details of `./project_management`
- `ORIGINAL_PLAN.md`
  - This is created by the user. Never edit this file. But user may update this. So, please check periodically.
  - `progress.md`
    - Progress management file in markdown format
  - `progress.mmd`
    - Progress management file in mermaid format
    - This should reflect the contents of `progress.md`


## General Python Rules
- Do not use try-except blocks as much as possible (makes debugging challenging with invisible errors)
- Ensure indent level matches with input
- Do not change header, footer, and these tags:
  - """Imports"""
  - """Parameters"""
  - """Functions & Classes"""
- Update placeholder of top-level docstring to provide overview of the file contents. This must be as simple as possible
  ``` python
  """
  1. Functionality:
     - (e.g., Executes XYZ operation)
  2. Input:
     - (e.g., Required data for XYZ)
  3. Output:
     - (e.g., Results of XYZ operation)
  4. Prerequisites:
     - (e.g., Necessary dependencies for XYZ)

  (Remove me: Please fill docstrings above, while keeping the bulette point style, and remove this instruction line)
  """
  ```

- Use `argparse` even when no arguments are necessary
  - Every python file should have at least one argument: -h|--help
- Always use `main` guard:
  ```python
  if __name__ == "__main__":
      main()
  ```
- Adopt modular approaches with smaller functions
- Implement Python logging (instead of print statements):
  ```python
  import logging
  logging.error()
  logging.warning() 
  logging.info()
  logging.debug()
  ```
  - Replace print statements: `print()` -> `logging.info()`
  - Progress tracking: 
    - `from tqdm import tqdm`
    - `progress_bar = tqdm(total=100)`
  - Debug tracing:
    - `from icecream import ic`
    - `ic()` for variable inspection

- Format integer numbers with underscores (e.g., 100_000)
- For string type of numbers, use commas by every three digits (e.g., "100,000")
- Fix random seed as 42

- Naming conventions:
  - Variable names in scripts:
    - lpath(s) - load paths
    - ldir - load directory
    - spath(s) - save paths
    - sdir - save directory
    - path(s) - generic paths
    - fname - file name
  - Directory names should be noun forms: e.g., `mnist-creation/`, `mnist-classification/`
  - Script file name should start with verb if it expects inputs and produces outputs: e.g, `classify_mnist.py`
  - Definition files for classes should be Capital: `ClassName.py`
  - Predefined parameters must be written in uppercase letters: `PARAM`

## Python Type Hints
- Explicitly define variable types in functions and classes
  ```python
  from typing import Union, Tuple, List, Dict, Any, Optional, Callable
  from collections.abc import Iterable
  ArrayLike = Union[List, Tuple, np.ndarray, pd.Series, pd.DataFrame, xr.DataArray, torch.Tensor]
  ```

## Python Docstring Format
- For Python functions, use NumPy-style docstrings
```python
def func(arg1: int, arg2: str) -> bool:
    """Summary line.

    Extended description of function.

    Example
    ----------
    >>> xx, yy = 1, "test"
    >>> out = func(xx, yy)
    >>> print(out)
    True

    Parameters
    ----------
    arg1 : int
        Description of arg1
    arg2 : str
        Description of arg2

    Returns
    -------
    bool
        Description of return value

    """
    return True
```
- But, one line docstring is also acceptable when function is simple
```python
def func(arg1: int, arg2: str) -> bool:
    """Summary line."""
    ...
    return arg1 + arg2
```

#### Top-level Docstring Format
- For Python scripts, use the following custom top-level docstring format
```python
"""
1. Functionality:
   - (e.g., Executes XYZ operation)
2. Input:
   - (e.g., Required data for XYZ)
3. Output:
   - (e.g., Results of XYZ operation)
4. Prerequisites:
   - (e.g., Necessary dependencies for XYZ)

(Remove me: Please fill docstrings above, while keeping the bullet point style, and remove this instruction line)
"""
```

## Python Function Hierarchy, Sorting Rules
- Functions must be sorted considering their hierarchy.
- Upstream functions should be placed in upper positions
  - from top (upstream functions) to down (utility functions)
- Do not change any code contents during sorting
- Includes comments to show hierarchy
- Use this kinds of separators as TWO LINES OF COMMENTING CODE in the file type
```python
# 1. Main entry point
# ---------------------------------------- 


# 2. Core functions
# ---------------------------------------- 


# 3. Helper functions
# ---------------------------------------- 
```

### Python Testing
- DO NOT USE UNITTEST BUT PYTEST
- Prepare pytest.ini
- Structure:
  - `./src/project-name/__init__.py` (for pip packages)
  - `./scripts/project-name/__init__.py` (for scientific projects)
  - `./tests/project-name/...`
  - `./tests/custom/...`
- Each test function should be smallest
- Each test file contain only one test function
- Each Test Class should be defined in a dedicated script
- !!! IMPORTANT !!! Test codes MUST BE MEANINGFUL, ATOMIC, and SELF-EXPLANATORY
  - Use `./run_tests.sh` in project root

### Python Test Script Example

``` python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-03 00:49:28 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/color/test__get_colors_from_cmap.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/color/test__get_colors_from_cmap.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import numpy as np
import pytest


def test_get_color_from_cmap():
    from mngs.plt.color._get_colors_from_cmap import get_color_from_cmap

    # Test with default parameters
    color = get_color_from_cmap("viridis", 0.5)
    assert isinstance(color, tuple)
    assert len(color) == 4  # RGBA
    assert all(0 <= val <= 1 for val in color)  # Values within range

    # Test with custom value range
    color_range = get_color_from_cmap("plasma", 10, value_range=(0, 20))
    assert isinstance(color_range, tuple)
    assert len(color_range) == 4

    # Test with custom alpha
    color_alpha = get_color_from_cmap("Blues", 0.7, alpha=0.5)
    assert color_alpha[3] == 0.5  # Alpha should be set correctly

    # Test value outside range gets clipped
    color_clipped = get_color_from_cmap("coolwarm", 2.0, value_range=(0, 1))
    color_expected = get_color_from_cmap("coolwarm", 1.0, value_range=(0, 1))
    assert color_clipped == color_expected


def test_get_colors_from_cmap():
    import matplotlib
    from mngs.plt.color._get_colors_from_cmap import get_colors_from_cmap

    # Test with default parameters
    colors = get_colors_from_cmap("viridis", 5)
    assert isinstance(colors, list)
    assert len(colors) == 5
    assert all(isinstance(color, tuple) for color in colors)
    assert all(len(color) == 4 for color in colors)  # Each color is RGBA

    # Test with custom value range
    colors_range = get_colors_from_cmap("plasma", 3, value_range=(-1, 1))
    assert len(colors_range) == 3

    # Test colors are evenly spaced
    colors_check = get_colors_from_cmap("viridis", 3)
    cmap = matplotlib.cm.get_cmap("viridis")

    # Calculate colors directly from cmap to compare
    expected_colors = [
        cmap(0.0),  # First color (0.0)
        cmap(0.5),  # Middle color (0.5)
        cmap(1.0),  # Last color (1.0)
    ]

    for idx, color in enumerate(colors_check):
        assert color == pytest.approx(expected_colors[idx], abs=1e-6)


def test_get_categorical_colors_from_cmap():
    from mngs.plt.color._get_colors_from_cmap import \
        get_categorical_colors_from_cmap

    # Test with list categories
    categories = ["cat", "dog", "bird"]
    cat_colors = get_categorical_colors_from_cmap("tab10", categories)
    assert isinstance(cat_colors, dict)
    assert set(cat_colors.keys()) == set(categories)
    assert all(len(color) == 4 for color in cat_colors.values())

    # Test with numpy array categories
    num_categories = np.array([1, 2, 3, 1, 2])
    num_cat_colors = get_categorical_colors_from_cmap(
        "viridis", num_categories, alpha=0.7
    )
    assert set(num_cat_colors.keys()) == {1, 2, 3}
    assert all(color[3] == 0.7 for color in num_cat_colors.values())

    # Test with duplicate categories
    duplicate_categories = ["A", "B", "A", "C", "B"]
    dup_cat_colors = get_categorical_colors_from_cmap(
        "plasma", duplicate_categories
    )
    assert set(dup_cat_colors.keys()) == {"A", "B", "C"}
    assert len(dup_cat_colors) == 3

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])


# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/color/_get_colors_from_cmap.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-02 17:53:27 (ywatanabe)"
# # File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/color/_get_from_cmap.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/plt/color/_get_from_cmap.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# from typing import List, Optional, Tuple, Union
# 
# import matplotlib
# import numpy as np
# 
# # class ColorGetter:
# #     # https://stackoverflow.com/questions/26108436/how-can-i-get-the-matplotlib-rgb-color-given-the-colormap-name-boundrynorm-an
# #     def __init__(self, cmap_name, start_val, stop_val):
# #         self.cmap_name = cmap_name
# #         self.cmap = plt.get_cmap(cmap_name)
# #         self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
# #         self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
# 
# #     def get_rgb(self, val):
# #         return self.scalarMap.to_rgba(val)
# 
# 
# def get_color_from_cmap(
#     cmap_name: str,
#     value: float,
#     value_range: Optional[Tuple[float, float]] = None,
#     alpha: float = 1.0,
# ) -> Tuple[float, float, float, float]:
#     """Get a color from a colormap at a specific value.
# 
#     Parameters
#     ----------
#     cmap_name : str
#         Name of the colormap (e.g., 'viridis', 'plasma', 'Blues')
#     value : float
#         Value to map to a color in the colormap
#     value_range : tuple of (float, float), optional
#         Range of values to map to the colormap. If None, uses (0, 1)
#     alpha : float, optional
#         Alpha value for the color (0.0 to 1.0), by default 1.0
# 
#     Returns
#     -------
#     tuple
#         RGBA color tuple with values from 0 to 1
#     """
#     # Get the colormap
#     cmap = matplotlib.cm.get_cmap(cmap_name)
# 
#     # Normalize the value
#     if value_range is None:
#         norm_value = value
#     else:
#         min_val, max_val = value_range
#         norm_value = (value - min_val) / (max_val - min_val)
# 
#     # Clip to ensure within range
#     norm_value = np.clip(norm_value, 0.0, 1.0)
# 
#     # Get the color
#     rgba_color = list(cmap(norm_value))
# 
#     # Set alpha
#     rgba_color[3] = alpha
# 
#     return tuple(rgba_color)
# 
# 
# def get_colors_from_cmap(
#     cmap_name: str,
#     n_colors: int,
#     value_range: Optional[Tuple[float, float]] = None,
#     alpha: float = 1.0,
# ) -> List[Tuple[float, float, float, float]]:
#     """Get a list of evenly spaced colors from a colormap.
# 
#     Parameters
#     ----------
#     cmap_name : str
#         Name of the colormap (e.g., 'viridis', 'plasma', 'Blues')
#     n_colors : int
#         Number of colors to sample from the colormap
#     value_range : tuple of (float, float), optional
#         Range of values to map to the colormap. If None, uses (0, 1)
#     alpha : float, optional
#         Alpha value for the colors (0.0 to 1.0), by default 1.0
# 
#     Returns
#     -------
#     list
#         List of RGBA color tuples with values from 0 to 1
#     """
#     if value_range is None:
#         values = np.linspace(0, 1, n_colors)
#     else:
#         values = np.linspace(value_range[0], value_range[1], n_colors)
# 
#     return [
#         get_color_from_cmap(cmap_name, val, value_range, alpha)
#         for val in values
#     ]
# 
# 
# def get_categorical_colors_from_cmap(
#     cmap_name: str, categories: Union[List, np.ndarray], alpha: float = 1.0
# ) -> dict:
#     """Map categorical values to colors from a colormap.
# 
#     Parameters
#     ----------
#     cmap_name : str
#         Name of the colormap (e.g., 'viridis', 'plasma', 'Blues')
#     categories : list or np.ndarray
#         List of categories to map to colors
#     alpha : float, optional
#         Alpha value for the colors (0.0 to 1.0), by default 1.0
# 
#     Returns
#     -------
#     dict
#         Dictionary mapping categories to RGBA color tuples
#     """
#     unique_categories = np.unique(categories)
#     n_categories = len(unique_categories)
# 
#     colors = get_colors_from_cmap(cmap_name, n_categories, alpha=alpha)
# 
#     return {cat: colors[idx] for idx, cat in enumerate(unique_categories)}
# 
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/color/_get_colors_from_cmap.py
# --------------------------------------------------------------------------------
```

## Your Understanding Check
Did you understand the guideline? If yes, please say:
`CLAUDE UNDERSTOOD: <THIS FILE PATH HERE>`

<!-- EOF -->