# MNGS Architecture Documentation

## Overview
MNGS (monogusa) is a comprehensive Python utility package for scientific computing, providing modules for data processing, visualization, machine learning, and more. This document describes the architecture, module relationships, and design principles.

## Module Dependency Analysis

### Summary
- **Total modules**: 26
- **Total dependencies**: 89
- **Circular dependencies**: 1 (in AI module - scheduled for refactoring)
- **Average coupling**: 3.4 dependencies per module

### Dependency Visualization
![Module Dependencies](../project_management/module_dependencies.png)

## Core Architecture

### 1. Foundation Layer (Low Coupling)
These modules have minimal dependencies and provide core functionality:

#### **types** (7 coupling)
- Provides type definitions (`ArrayLike`, `ColorLike`)
- Used by: ai, pd, plt, stats
- Dependencies: Basic type checking utilities

#### **dict** (5 coupling)
- Dictionary utilities and DotDict implementation
- Used by: gen, io, plt, str
- Dependencies: utils

#### **str** (7 coupling)
- String manipulation and formatting utilities
- Used by: db, decorators, dsp, gen, io, resource
- Dependencies: dict

### 2. Utility Layer (Medium Coupling)

#### **decorators** (22 coupling) - High coupling, needs attention
- Function decorators for type conversion, caching, etc.
- Used by: dsp, gen, io, linalg, nn, plt, stats
- Dependencies: Multiple utility modules

#### **path** (2 coupling)
- Path manipulation utilities
- Used by: gen, io
- No outgoing dependencies

#### **utils** (4 coupling)
- General utilities
- Used by: dict, gen, plt
- Dependencies: reproduce

### 3. Data Processing Layer

#### **pd** (2 coupling)
- Pandas utilities and extensions
- Used by: plt
- Dependencies: types

#### **dsp** (19 coupling) - High coupling
- Digital signal processing
- Used by: nn
- Dependencies: decorators, gen, io, nn, str

#### **stats** (10 coupling)
- Statistical analysis tools
- No incoming dependencies
- Dependencies: decorators, tests, types

### 4. I/O and Storage Layer

#### **io** (28 coupling) - Highest coupling, needs refactoring
- File I/O operations for multiple formats
- Used by: ai, dsp, gen, plt, resource
- Dependencies: Many internal modules

#### **db** (2 coupling)
- Database operations
- Used by: io
- Dependencies: str

### 5. Visualization Layer

#### **plt** (11 coupling)
- Plotting and visualization
- Used by: gen, tex
- Dependencies: context, decorators, dict, io, pd, types, utils

### 6. Machine Learning Layer

#### **ai** (4 coupling)
- AI/ML utilities and GenAI providers
- No incoming dependencies
- Dependencies: _gen_ai, io, reproduce, types
- Contains circular dependency (to be fixed)

#### **nn** (20 coupling) - High coupling
- Neural network layers and utilities
- Used by: dsp
- Dependencies: Multiple processing modules

### 7. Application Layer

#### **gen** (15 coupling)
- General utilities and workflow management
- Used by: __main__, dsp, nn
- Dependencies: Multiple utility modules

#### **resource** (3 coupling)
- System resource monitoring
- No incoming dependencies
- Dependencies: _sh, io, str

## Circular Dependencies

### Identified Circular Dependencies:
1. **AI Module**: `mngs.ai._gen_ai._genai_factory` → `mngs.ai._gen_ai._Perplexity` → `mngs.ai._gen_ai._genai_factory`
   - **Status**: Scheduled for refactoring in AI module refactoring task
   - **Solution**: Implement proper factory pattern without circular imports

## Module Interfaces

### Core Module APIs

#### 1. **io Module**
Primary interface for file operations:
```python
# Main functions
mngs.io.save(data, path)
mngs.io.load(path)
mngs.io.glob(pattern)
```

#### 2. **gen Module**
Workflow and experiment management:
```python
# Main functions
mngs.gen.start(sys, sdir, exp=None)
mngs.gen.close(CONFIG)
```

#### 3. **plt Module**
Enhanced plotting interface:
```python
# Main functions
fig, axes = mngs.plt.subplots()
mngs.plt.configure_mpl(CONFIG)
```

#### 4. **dsp Module**
Signal processing utilities:
```python
# Main functions
mngs.dsp.demo_sig(n_chs, samp_rate, sig_len_sec)
mngs.dsp.filt(data, low_hz, high_hz, samp_rate)
```

## Design Principles

### 1. **Modularity**
- Each module should have a clear, focused purpose
- Minimize inter-module dependencies
- Use dependency injection where possible

### 2. **Extensibility**
- Decorators provide cross-cutting functionality
- Plugin-style architecture for file formats (io module)
- Abstract base classes for providers (ai module)

### 3. **Type Safety**
- Centralized type definitions in types module
- Type hints throughout the codebase
- Runtime type checking via decorators

### 4. **Performance**
- Lazy imports where appropriate
- Caching decorators for expensive operations
- Batch operations support

## Recommendations for Improvement

### High Priority
1. **Refactor io module** (28 coupling)
   - Split into smaller, focused submodules
   - Create clear interfaces for file format handlers
   - Reduce dependencies on internal modules

2. **Refactor AI module**
   - Fix circular dependency
   - Implement proper factory pattern
   - Separate concerns (providers, utilities, costs)

3. **Reduce decorators coupling** (22 coupling)
   - Consider splitting into multiple decorator modules
   - Reduce dependencies on specific data types

### Medium Priority
1. **Refactor nn module** (20 coupling)
   - Consider splitting DSP-specific layers
   - Create clearer layer categories

2. **Refactor dsp module** (19 coupling)
   - Reduce dependency on nn module
   - Create clearer submodule structure

### Low Priority
1. **Document module interfaces**
   - Create interface documentation for each module
   - Define public vs private APIs
   - Add usage examples

2. **Create module guidelines**
   - Define when to create new modules
   - Guidelines for managing dependencies
   - Best practices for module design

## Module Dependency Matrix

| From/To | ai | db | decorators | dsp | gen | io | nn | pd | plt | stats | str | types | utils |
|---------|----|----|------------|-----|-----|----|----|----|-----|-------|-----|-------|--------|
| ai      | -  | 0  | 0          | 0   | 0   | 1  | 0  | 0  | 0   | 0     | 0   | 1     | 0      |
| db      | 0  | -  | 0          | 0   | 0   | 0  | 0  | 0  | 0   | 0     | 1   | 0     | 0      |
| decorators| 0| 0  | -          | 0   | 0   | 0  | 0  | 0  | 0   | 0     | 1   | 0     | 0      |
| dsp     | 0  | 0  | 1          | -   | 1   | 1  | 1  | 0  | 0   | 0     | 1   | 0     | 0      |
| gen     | 0  | 0  | 1          | 0   | -   | 1  | 0  | 0  | 1   | 0     | 1   | 0     | 1      |
| io      | 0  | 1  | 1          | 0   | 0   | -  | 0  | 0  | 0   | 0     | 1   | 0     | 0      |
| nn      | 0  | 0  | 1          | 1   | 1   | 0  | -  | 0  | 0   | 0     | 0   | 0     | 0      |
| pd      | 0  | 0  | 0          | 0   | 0   | 0  | 0  | -  | 0   | 0     | 0   | 1     | 0      |
| plt     | 0  | 0  | 1          | 0   | 0   | 1  | 0  | 1  | -   | 0     | 0   | 1     | 1      |
| stats   | 0  | 0  | 1          | 0   | 0   | 0  | 0  | 0  | 0   | -     | 0   | 1     | 0      |

## Conclusion

MNGS has a well-structured architecture with clear module separation. The main areas for improvement are:
1. Reducing coupling in io, decorators, nn, and dsp modules
2. Fixing the circular dependency in the AI module
3. Creating clearer module interfaces and documentation

The ongoing AI module refactoring (10-15 day project) will address one of the key architectural issues. Future work should focus on the highly coupled io module and establishing clearer architectural guidelines.