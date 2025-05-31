# AI Module Refactoring - Quick Reference for Agents

## 🚀 Quick Start
1. Check BULLETIN-BOARD.md for latest status
2. Find your assigned phase below
3. Read detailed task in AI_MODULE_REFACTORING_TASKS.md
4. Post updates to bulletin board

## 📋 Phase Overview

### Phase 1: Architecture (Days 1-3)
```bash
# Key files to modify
src/mngs/ai/optim/Ranger_Deep_Learning_Optimizer/  # Extract this
src/mngs/ai/*.py                                    # Rename camelCase files
src/mngs/ai/__init__.py                            # Update imports
```

### Phase 2: GenAI (Days 4-6)
```bash
# Focus area
src/mngs/ai/_gen_ai/_BaseGenAI.py      # Break down god object
src/mngs/ai/_gen_ai/_genai_factory.py  # Type-safe factory
src/mngs/ai/_gen_ai/_*.py              # Standardize providers
```

### Phase 3: Testing (Days 7-9)
```bash
# Create tests
tests/mngs/ai/genai/     # New test structure
tests/mngs/ai/fixtures/  # Mock API responses
tests/mngs/ai/integration/  # End-to-end tests
```

### Phase 4: Integration (Days 10-12)
```bash
# Update areas
examples/mngs/ai/        # Update examples
docs/api/mngs.ai.*.rst   # Update API docs
MIGRATION_GUIDE.md       # Create this
```

## 🔑 Key Principles

1. **No Breaking Changes Without Migration Path**
2. **Maintain 100% Test Coverage**
3. **Follow MNGS Coding Style**
4. **Document Everything**

## 📝 Bulletin Board Template
```markdown
## Agent: [Your Name]-[Time]
Role: [Your Phase] Specialist
Status: working on
Task: [Current specific task]
Notes:
- [Progress update]
- [Any blockers]
- [Next steps]
@mentions: [Tag relevant agents]
Timestamp: [YYYY-MM-DD-HH:MM]
```

## ⚡ Common Commands
```bash
# Run tests for AI module
pytest tests/mngs/ai/ -v

# Check test coverage
pytest tests/mngs/ai/ --cov=mngs.ai --cov-report=html

# Format code
black src/mngs/ai/
isort src/mngs/ai/

# Type check
mypy src/mngs/ai/
```

## 🚨 Blockers? 
1. Post to BULLETIN-BOARD.md with @mentions
2. Check feature request for clarification
3. Refer to existing module patterns (gen, io, plt)

## ✅ Definition of Done
- [ ] Code implemented and formatted
- [ ] Tests written and passing
- [ ] Documentation updated
- [ ] Examples working
- [ ] Bulletin board updated
- [ ] Next agent notified

---
**Remember**: We're refactoring because the user "was not experienced at the time of development". Let's make it excellent! 🎯