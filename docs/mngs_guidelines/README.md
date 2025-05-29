# MNGS Documentation

## 🚀 Quick Links

- [Why Use MNGS?](agent_guidelines/00_why_use_mngs.md)
- [5-Minute Quick Start](agent_guidelines/01_quick_start.md)
- [Core Concepts](agent_guidelines/02_core_concepts.md)
- [Module Overview](agent_guidelines/03_module_overview.md)
- [Common Workflows](agent_guidelines/04_common_workflows.md)

## 📚 Module Documentation

### Core Modules
- [mngs.io](modules/io/README.md) - Universal file I/O
- [mngs.gen](modules/gen/README.md) - General utilities and environment setup
- [mngs.plt](modules/plt/README.md) - Enhanced plotting with data tracking

### Data Processing
- [mngs.dsp](modules/dsp/README.md) - Digital signal processing
- [mngs.pd](modules/pd/README.md) - Pandas utilities
- [mngs.stats](modules/stats/README.md) - Statistical analysis

### Machine Learning
- [mngs.ai](modules/ai/README.md) - AI/ML utilities
- [mngs.nn](modules/nn/README.md) - Neural network layers
- [mngs.torch](modules/torch/README.md) - PyTorch utilities

### Utilities
- [mngs.path](modules/path/README.md) - Path operations
- [mngs.str](modules/str/README.md) - String utilities
- [mngs.decorators](modules/decorators/README.md) - Function decorators

## 🎯 For Agents

This documentation is designed to be agent-friendly:

1. **Structured Format**: Consistent documentation structure across all modules
2. **Code Examples**: Every function includes practical examples
3. **Searchable**: Use standard markdown headings for easy navigation
4. **Complete API Reference**: All functions and parameters documented

### How to Search

To find information about a specific function:
```
grep -r "function_name" docs/mngs_guidelines/
```

To find examples of a specific use case:
```
grep -r "use_case" docs/mngs_guidelines/modules/*/examples/
```

## 📖 Documentation Structure

```
docs/mngs_guidelines/
├── README.md                    # This file
├── agent_guidelines/            # High-level guides
│   ├── 00_why_use_mngs.md     # Motivation and benefits
│   ├── 01_quick_start.md       # 5-minute introduction
│   ├── 02_core_concepts.md     # Key concepts
│   ├── 03_module_overview.md   # Module descriptions
│   └── 04_common_workflows.md  # Typical usage patterns
├── modules/                     # Detailed module docs
│   ├── io/
│   │   ├── README.md           # Module overview
│   │   ├── load.md             # Function details
│   │   ├── save.md             # Function details
│   │   └── examples/           # Code examples
│   └── [other modules...]
├── api_reference/              # Auto-generated API docs
└── tutorials/                  # Step-by-step tutorials
```

## 🔄 Documentation Status

### Completed
- ✅ Agent guidelines (all 5 documents)
- ✅ mngs.io module documentation
- ✅ Basic structure setup

### In Progress
- 🚧 Module-specific documentation
- 🚧 API reference generation
- 🚧 Example scripts

### TODO
- ⏳ Sphinx configuration
- ⏳ Interactive tutorials
- ⏳ Video guides

## 🛠️ Contributing to Documentation

When adding new documentation:

1. Follow the existing structure
2. Include code examples for every function
3. Use consistent formatting
4. Test all code examples
5. Update this index

## 📞 Getting Help

- **Issues**: Report at [GitHub Issues](https://github.com/ywatanabe1989/mngs/issues)
- **Source**: View at [GitHub Repository](https://github.com/ywatanabe1989/mngs)
- **Examples**: Check the `examples/` directory in the repository

## Version

This documentation is for mngs version 1.9.2+

Last updated: 2025-05-30