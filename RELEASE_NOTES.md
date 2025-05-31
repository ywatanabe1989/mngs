# Release Notes - v1.0.0

## 🎉 First Major Release

We're excited to announce the first major release of MNGS (Monogusa)! This release represents a significant milestone with comprehensive testing, documentation, and CI/CD infrastructure.

## ✨ Major Achievements

### 📊 100% Test Coverage
- **118 unit tests** covering all modules
- **10 integration tests** verifying cross-module functionality
- Comprehensive test suite ensuring reliability

### 📚 Complete Documentation
- Full API documentation for all modules
- Module-specific README files with examples
- Sphinx documentation setup with RTD theme
- Getting started guide and installation instructions

### 🔧 Production-Ready Infrastructure
- GitHub Actions CI/CD pipeline
- Automated testing on Python 3.8-3.11
- Pre-commit hooks for code quality
- Automated release workflow

### 🧩 Module Highlights

#### 🧠 AI Module
- Generative AI interfaces (OpenAI, Anthropic, Google, etc.)
- Classification tools and metrics
- Feature extraction and clustering utilities

#### 📊 DSP Module
- Signal filtering and processing
- Phase-amplitude coupling (PAC) analysis
- Wavelet transforms and spectral analysis
- Ripple detection algorithms

#### 🗄️ DB Module
- PostgreSQL and SQLite3 support
- Comprehensive database operations
- Transaction management
- Import/export utilities

#### 📈 Stats Module
- Statistical tests with multiple comparison corrections
- Correlation analysis
- Descriptive statistics with NaN handling

#### 🎨 PLT Module
- Enhanced matplotlib plotting
- Automatic axis formatting
- Color management utilities
- Figure export capabilities

#### 🔄 Other Modules
- **gen**: General utilities and experiment management
- **io**: Flexible file I/O with multiple format support
- **pd**: Enhanced pandas operations
- **nn**: Neural network layers for signal processing
- **path**: Advanced path manipulation

## 🚀 Installation

```bash
pip install mngs
```

## 📖 Quick Start

```python
import mngs

# Enhanced plotting
fig, ax = mngs.plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
fig.savefig("output.png")

# Signal processing
data = mngs.dsp.demo_sig(n_chs=3)
filtered = mngs.dsp.filt.bandpass(data, 1, 50)

# File I/O
mngs.io.save(data, "data.pkl")
loaded = mngs.io.load("data.pkl")
```

## 🙏 Acknowledgments

Special thanks to all contributors and users who have helped shape this release. Your feedback and support have been invaluable.

## 📝 What's Next

We're committed to continuous improvement. Future releases will focus on:
- Performance optimizations
- Additional module features
- Extended documentation and tutorials
- Community-requested enhancements

For the complete list of changes, see [CHANGELOG.md](CHANGELOG.md).

---

**Full Changelog**: https://github.com/ywatanabe1989/mngs/releases/tag/v1.0.0