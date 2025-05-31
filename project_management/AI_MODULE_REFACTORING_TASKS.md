<!-- ---
!-- Timestamp: 2025-05-31 04:42:46
!-- Author: ywatanabe
!-- File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/.claude-worktree/mngs_repo/project_management/AI_MODULE_REFACTORING_TASKS.md
!-- --- -->

# AI Module Refactoring - Agent Task Assignments

## Overview
The AI module requires comprehensive refactoring due to architectural debt from early development. This document provides specific task assignments for multiple agents to coordinate the refactoring effort.

**Tasks**:
1. **Extract Ranger Optimizer** (Day 1)
   - Move `mngs/ai/optim/Ranger_Deep_Learning_Optimizer/` to external dependency
   - Create pip installable package or use existing ranger21 package
   - Update imports throughout codebase
   - File: `src/mngs/ai/optim/`

<!-- CLASSES AND THEIR DEFINITIONS FILES MUST BE CAMEL CASE
 !-- 2. **Standardize Naming Conventions** (Day 1-2)
 !--    - Convert all camelCase to snake_case:
 !--      - `ClassificationReporter` → `classification_reporter`
 !--      - `ClassifierServer` → `classifier_server`
 !--      - `EarlyStopping` → `early_stopping`
 !--      - `LearningCurveLogger` → `learning_curve_logger`
 !--    - Update all imports and references -->

3. **Create Module Structure** (Day 2-3)
   - Reorganize into clear submodules:
     ```
     mngs/ai/
     ├── genai/          # Generative AI providers
     ├── clustering/     # UMAP, PCA, etc.
     ├── metrics/        # bACC, silhouette score
     ├── training/       # EarlyStopping, LearningCurveLogger
     ├── visualization/  # plt submodule
     └── sklearn/        # sklearn wrappers
     ```

### Agent 2: GenAI Module Specialist
**Timeline**: Days 4-6  
**Priority**: HIGH  
**Dependencies**: Agent 1 completion

**Tasks**:
1. **Refactor BaseGenAI** (Day 4)
   - Break down god object into focused components
   - Separate concerns: auth, requests, formatting, costs
   - Create abstract base class with clear interface
   - File: `src/mngs/ai/_gen_ai/_BaseGenAI.py`

2. **Implement Strategy Pattern** (Day 5)
   - Create provider strategy interface
   - Refactor each provider to implement interface
   - Remove code duplication between providers
   - Files: All in `src/mngs/ai/_gen_ai/`

3. **Type-Safe Factory** (Day 6)
   - Replace string-based factory with enum
   - Add comprehensive type hints
   - File: `src/mngs/ai/_gen_ai/_genai_factory.py`

### Agent 3: Testing Specialist
**Timeline**: Days 7-9  
**Priority**: HIGH  
**Dependencies**: Agent 2 completion

**Tasks**:
1. **Create Test Structure** (Day 7)
   - Mirror new module structure in tests
   - Create test fixtures for mocking APIs
   - Set up test data and expected outputs

2. **Implement Unit Tests** (Day 8)
   - Test each provider with mocked API calls
   - Test factory pattern
   - Test utility functions
   - Achieve 100% coverage target

3. **Integration Tests** (Day 9)
   - Test provider switching
   - Test error handling
   - Test cost calculations
   - Test with real API calls (optional, with keys)

### Agent 4: Integration Specialist
**Timeline**: Days 10-12  
**Priority**: MEDIUM  
**Dependencies**: Agent 3 completion

**Tasks**:
1. **Update Dependencies** (Day 10)
   - Find all code using old AI module structure
   - Update imports and function calls
   - Ensure backward compatibility where needed

2. **Migration Guide** (Day 11)
   - Create detailed migration documentation
   - Show before/after code examples
   - List breaking changes
   - Provide compatibility shims if needed

3. **Update Documentation** (Day 12)
   - Update all examples using AI module
   - Update API documentation
   - Update tutorials and guides
   - Verify all docs build correctly

### Agent 5: Documentation Specialist
**Timeline**: Parallel with other agents  
**Priority**: MEDIUM  
**Dependencies**: Understanding of refactoring plan

**Tasks**:
1. **Create AI Module Guide**
   - File: `docs/mngs_guidelines/modules/IMPORTANT-MNGS-22-ai-module-detailed.md`
   - Follow pattern of gen/io module guides
   - Include all providers and utilities
   - Add troubleshooting section

2. **Create NN Module Guide**
   - File: `docs/mngs_guidelines/modules/IMPORTANT-MNGS-23-nn-module-detailed.md`
   - Document all neural network layers
   - Include usage examples
   - Add integration with PyTorch

3. **Update Complete Reference**
   - Update `MNGS_COMPLETE_REFERENCE.md` after refactoring
   - Ensure all new APIs are documented
   - Add deprecation notices for old APIs

## Communication Protocol

1. **Daily Updates**: Each agent posts progress to BULLETIN-BOARD.md
2. **Blocking Issues**: Tag relevant agents in bulletin board
3. **Completion**: Mark phase complete and notify next agent
4. **Questions**: Use @mentions in bulletin board

## Success Criteria

- [ ] All tests passing (100% coverage)
- [ ] No breaking changes without migration path
- [ ] Documentation complete and accurate
- [ ] Examples updated and working
- [ ] Performance benchmarks maintained
- [ ] Code follows MNGS style guidelines

## Risk Mitigation

1. **Backward Compatibility**: Create compatibility layer if needed
2. **API Changes**: Provide clear migration path
3. **Performance**: Benchmark before/after refactoring
4. **Dependencies**: Test with different Python versions

## Resources

- Feature Request: `/project_management/feature_requests/feature-request-ai-module-refactoring.md`
- Current AI Module: `/src/mngs/ai/`
- Test Directory: `/tests/mngs/ai/`
- Documentation: `/docs/mngs_guidelines/`

---

**Note**: This is a living document. Agents should update task status and add notes as work progresses.

<!-- EOF -->