<!-- ---
!-- Timestamp: 2025-05-03 16:58:07
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/mngs_repo/README.md
!-- --- -->


# mngs (monogusa; meaning lazy person in Japanese)
A Python utility package for simplifying common research and development tasks.

<!-- badges -->
[![PyPI version](https://badge.fury.io/py/mngs.svg)](https://badge.fury.io/py/mngs)
![CI](https://github.com/ywatanabe1989/mngs/actions/workflows/install-pypi-latest.yml/badge.svg)
![CI](https://github.com/ywatanabe1989/mngs/actions/workflows/install-latest-release.yml/badge.svg)
![CI](https://github.com/ywatanabe1989/mngs/actions/workflows/install-develop-branch.yml/badge.svg)
![CI](https://github.com/ywatanabe1989/mngs/actions/workflows/custom-run-pytest.yml/badge.svg)

<!-- $ sudo apt-get install libportaudio2 -->
## Installation
```bash
# 1. From pypi.org
$ pip install mngs

# 2. From git (direct)
$ pip install git+https://github.com/ywatanabe1989/mngs.git@develop

# 3. From git develop
$ git clone git clone git@github.com:ywatanabe1989/mngs.git mngs_dev cd mngs_dev && 
$ python3.11 -m venv .env && source .env/bin/activate && python -m pip install -e .
```


## Submodules

| Category              | Submodule                                         | Description                      |
|-----------------------|---------------------------------------------------|----------------------------------|
| **Fundamentals**      | [`mngs.gen`](./src/mngs/gen#readme)               | General utilities                |
|                       | [`mngs.io`](./src/mngs/io#readme)                 | Input/Output operations          |
|                       | [`mngs.utils`](./src/mngs/utils#readme)           | General utilities                |
|                       | [`mngs.dict`](./src/mngs/dict#readme)             | Dictionary utilities             |
|                       | [`mngs.str`](./src/mngs/str#readme)               | String manipulation              |
|                       | [`mngs.torch`](./src/mngs/torch#readme)           | PyTorch utilities                |
| **Data Science**      | [`mngs.plt`](./src/mngs/plt#readme)               | Plotting with automatic tracking |
|                       | [`mngs.stats`](./src/mngs/stats#readme)           | Statistical analysis             |
|                       | [`mngs.pd`](./src/mngs/pd#readme)                 | Pandas utilities                 |
|                       | [`mngs.tex`](./src/mngs/tex#readme)               | LaTeX utilities                  |
| **AI: ML/PR**         | [`mngs.ai`](./src/mngs/ai#readme)                 | AI and Machine Learning          |
|                       | [`mngs.nn`](./src/mngs/nn#readme)                 | Neural Networks                  |
|                       | [`mngs.torch`](./src/mngs/torch#readme)           | PyTorch utilities                |
|                       | [`mngs.db`](./src/mngs/db#readme)                 | Database operations              |
|                       | [`mngs.linalg`](./src/mngs/linalg#readme)         | Linear algebra                   |
| **Signal Processing** | [`mngs.dsp`](./src/mngs/dsp#readme)               | Digital Signal Processing        |
| **Statistics**        | [`mngs.stats`](./src/mngs/stats#readme)           | Statistical analysis tools       |
| **ETC**               | [`mngs.decorators`](./src/mngs/decorators#readme) | Function decorators              |
|                       | [`mngs.gists`](./src/mngs/gists#readme)           | Code snippets                    |
|                       | [`mngs.resource`](./src/mngs/resource#readme)     | Resource management              |
|                       | [`mngs.web`](./src/mngs/web#readme)               | Web-related functions            |

<!-- ## Submodules
 !-- #### Fundamentals
 !-- - [`mngs.gen`](./src/mngs/gen#readme): General utilities
 !-- - [`mngs.io`](./src/mngs/io#readme): Input/Output operations
 !-- - [`mngs.utils`](./src/mngs/utils#readme): General utilities
 !-- - [`mngs.dict`](./src/mngs/dict#readme): Dictionary utilities
 !-- - [`mngs.str`](./src/mngs/str#readme): String manipulation
 !-- - [`mngs.torch`](./src/mngs/torch#readme): PyTorch utilities
 !-- 
 !-- #### Data Science
 !-- - [`mngs.plt`](./src/mngs/plt#readme): Plotting with automatic tracking
 !-- - [`mngs.stats`](./src/mngs/stats#readme): Statistical analysis
 !-- - [`mngs.pd`](./src/mngs/pd#readme): Pandas utilities- 
 !-- - [`mngs.tex`](./src/mngs/tex#readme): LaTeX utilities
 !-- 
 !-- #### AI: Machine Learning and Pattern Recognition
 !-- - [`mngs.ai`](./src/mngs/ai#readme): AI and Machine Learning
 !-- - [`mngs.nn`](./src/mngs/nn#readme): Neural Networks
 !-- - [`mngs.torch`](./src/mngs/torch#readme): PyTorch utilities
 !-- - [`mngs.db`](./src/mngs/db#readme): Database operations
 !-- - [`mngs.linalg`](./src/mngs/linalg#readme): Linear algebra
 !-- 
 !-- #### Signal Processing
 !-- - [`mngs.dsp`](./src/mngs/dsp#readme): Digital Signal Processing
 !-- 
 !-- #### Statistics
 !-- - [`mngs.stats`](./src/mngs/stats#readme): Statistical analysis tools
 !-- 
 !-- #### ETC
 !-- - [`mngs.decorators`](./src/mngs/decorators#readme): Function decorators
 !-- - [`mngs.gists`](./src/mngs/gists#readme): Code snippets
 !-- - [`mngs.resource`](./src/mngs/resource#readme): Resource management
 !-- - [`mngs.web`](./src/mngs/web#readme): Web-related functions -->

## Modules and functions list
[`./docs/mngs_modules.csv`](./docs/mngs_modules.csv)

<!-- ## Documentation
 !-- For detailed documentation, please visit our [GitHub Pages](https://ywatanabe1989.github.io/mngs/). -->

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Contact
Yusuke Watanabe (ywatanabe@alumni.u-tokyo.ac.jp)

<!-- EOF -->