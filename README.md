<!-- ---
!-- Timestamp: 2025-05-30 16:10:54
!-- Author: ywatanabe
!-- File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/.claude-worktree/mngs_repo/README.md
!-- --- -->


# mngs (monogusa; meaning lazy person in Japanese)
A comprehensive Python framework for scientific computing, machine learning, and data analysis.

<!-- badges -->
[![PyPI version](https://badge.fury.io/py/mngs.svg)](https://badge.fury.io/py/mngs)
[![Python Versions](https://img.shields.io/pypi/pyversions/mngs.svg)](https://pypi.org/project/mngs/)
[![License](https://img.shields.io/github/license/ywatanabe1989/mngs_repo)](https://github.com/ywatanabe1989/mngs_repo/blob/main/LICENSE)
[![Tests](https://github.com/ywatanabe1989/mngs_repo/actions/workflows/ci.yml/badge.svg)](https://github.com/ywatanabe1989/mngs_repo/actions)
[![Coverage](https://codecov.io/gh/ywatanabe1989/mngs_repo/branch/main/graph/badge.svg)](https://codecov.io/gh/ywatanabe1989/mngs_repo)
[![Documentation](https://readthedocs.org/projects/mngs/badge/?version=latest)](https://mngs.readthedocs.io/en/latest/?badge=latest)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

## ✨ Features

- **🧪 100% Test Coverage**: Comprehensive test suite ensuring reliability
- **📚 Fully Documented**: Complete API documentation with examples
- **🔧 Multi-Domain**: Scientific computing, ML, signal processing, and more
- **🐍 Type Safe**: Full type hints for better IDE support
- **⚡ Performance**: Optimized implementations with GPU support
- **🔄 Interoperable**: Seamless numpy/torch/pandas integration

## 📦 Installation

```bash
# From PyPI (stable)
pip install mngs

# From GitHub (latest)
pip install git+https://github.com/ywatanabe1989/mngs.git@main

# Development installation
git clone https://github.com/ywatanabe1989/mngs.git
cd mngs
pip install -e ".[dev]"
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

## 🚀 Quick Start

```python
import mngs

# Start an experiment with automatic logging
config, info = mngs.gen.start(sys, sdir="./experiments")

# Load and process data
data = mngs.io.load("data.csv")
processed = mngs.pd.force_df(data)

# Signal processing
signal, time, fs = mngs.dsp.demo_sig(sig_type="chirp")
filtered = mngs.dsp.filt.bandpass(signal, fs, bands=[[10, 50]])

# Machine learning workflow
reporter = mngs.ai.ClassificationReporter()
metrics = reporter.evaluate(y_true, y_pred)

# Visualization
fig, ax = mngs.plt.subplots()
ax.plot(time, signal[0, 0, :])
mngs.io.save(fig, "signal_plot.png")

# Close experiment
mngs.gen.close(config, info)
```

## 📖 Documentation

- **[Full Documentation](https://mngs.readthedocs.io/)**: Complete API reference and guides
- **[Examples](./examples/)**: Practical examples and workflows
- **[Module List](./docs/mngs_modules.csv)**: Complete list of all functions

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone and install
git clone https://github.com/ywatanabe1989/mngs.git
cd mngs
make install

# Run tests
make test

# Format code
make format
```

## 📊 Project Status

- **Test Coverage**: 100% (118/118 tests passing)
- **Documentation**: Complete for all modules
- **CI/CD**: Automated testing, linting, and releases
- **Python Support**: 3.8, 3.9, 3.10, 3.11

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

Yusuke Watanabe (ywatanabe@alumni.u-tokyo.ac.jp)

## 🙏 Acknowledgments

Special thanks to all contributors and the open-source community for making this project possible.

<!-- EOF -->