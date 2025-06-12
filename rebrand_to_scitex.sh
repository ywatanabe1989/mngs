#!/bin/bash
# -*- coding: utf-8 -*-
# Quick rebranding script: mngs → scitex
# For personal use - simple and direct

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting rebranding: mngs → scitex${NC}"

# 1. Create backup tag
echo -e "${YELLOW}Creating backup tag...${NC}"
git tag -a "v1.11.0-final-mngs" -m "Final version before rebranding to scitex" || true
git push origin "v1.11.0-final-mngs" 2>/dev/null || true

# 2. Run the rename script
echo -e "${YELLOW}Running rename script...${NC}"
if [ -f "./docs/to_claude/bin/general/rename.sh" ]; then
    # First do a dry run
    echo -e "${YELLOW}Dry run to see changes:${NC}"
    ./docs/to_claude/bin/general/rename.sh 'mngs' 'scitex' . | head -20
    
    echo -e "\n${YELLOW}Press Enter to continue with actual rename, or Ctrl+C to cancel${NC}"
    read
    
    # Actual rename
    ./docs/to_claude/bin/general/rename.sh -n 'mngs' 'scitex' .
else
    echo -e "${RED}Rename script not found!${NC}"
    exit 1
fi

# 3. Rename the main source directory
echo -e "${YELLOW}Renaming source directory...${NC}"
if [ -d "src/mngs" ]; then
    mv src/mngs src/scitex
    echo -e "${GREEN}Renamed src/mngs → src/scitex${NC}"
fi

# 4. Update any remaining imports that might have been missed
echo -e "${YELLOW}Fixing any remaining imports...${NC}"
find . -name "*.py" -type f -not -path "./.git/*" -not -path "*/.*" | while read -r file; do
    # Use sed to replace imports
    sed -i.bak 's/from mngs/from scitex/g' "$file" && rm -f "$file.bak"
    sed -i.bak 's/import mngs/import scitex/g' "$file" && rm -f "$file.bak"
    sed -i.bak 's/mngs\./scitex./g' "$file" && rm -f "$file.bak"
done

# 5. Update pyproject.toml
echo -e "${YELLOW}Updating pyproject.toml...${NC}"
if [ -f "pyproject.toml" ]; then
    sed -i.bak 's/name = "mngs"/name = "scitex"/g' pyproject.toml && rm -f pyproject.toml.bak
    sed -i.bak 's/mngs/scitex/g' pyproject.toml && rm -f pyproject.toml.bak
    echo -e "${GREEN}Updated pyproject.toml${NC}"
fi

# 6. Update setup.py if it exists
if [ -f "setup.py" ]; then
    echo -e "${YELLOW}Updating setup.py...${NC}"
    sed -i.bak 's/name="mngs"/name="scitex"/g' setup.py && rm -f setup.py.bak
    sed -i.bak 's/mngs/scitex/g' setup.py && rm -f setup.py.bak
fi

# 7. Update README.md
echo -e "${YELLOW}Updating README.md...${NC}"
if [ -f "README.md" ]; then
    sed -i.bak 's/mngs/scitex/g' README.md && rm -f README.md.bak
    sed -i.bak 's/MNGS/SciTeX/g' README.md && rm -f README.md.bak
    echo -e "${GREEN}Updated README.md${NC}"
fi

# 8. Update __init__.py files to fix any module names
echo -e "${YELLOW}Updating module references...${NC}"
find . -name "__init__.py" -type f -not -path "./.git/*" | while read -r file; do
    sed -i.bak 's/__name__ = "mngs/__name__ = "scitex/g' "$file" && rm -f "$file.bak"
done

# 9. Run a quick test to see if imports work
echo -e "${YELLOW}Testing import...${NC}"
python -c "import sys; sys.path.insert(0, 'src'); import scitex; print('✓ Import successful!')" || {
    echo -e "${RED}Import test failed! Check for errors.${NC}"
}

# 10. Show summary of changes
echo -e "\n${GREEN}=== Rebranding Summary ===${NC}"
echo -e "1. All 'mngs' references updated to 'scitex'"
echo -e "2. Source directory renamed: src/mngs → src/scitex"
echo -e "3. Package name updated in pyproject.toml"
echo -e "4. README.md updated"

echo -e "\n${YELLOW}Next steps:${NC}"
echo -e "1. Run tests: ${GREEN}pytest tests/${NC}"
echo -e "2. Commit changes: ${GREEN}git add -A && git commit -m 'Rebrand: mngs → scitex'${NC}"
echo -e "3. Update git repo name on GitHub (optional)"
echo -e "4. When ready for PyPI: ${GREEN}python -m build && python -m twine upload dist/*${NC}"

echo -e "\n${GREEN}Rebranding complete!${NC}"

# Optional: Create a simple alias file for transition period
cat > ~/.mngs_to_scitex_alias.py << 'EOF'
# Temporary alias for transition - add to your scripts if needed
import sys
try:
    import scitex
    sys.modules['mngs'] = scitex
except ImportError:
    pass
EOF

echo -e "\n${YELLOW}Created ~/.mngs_to_scitex_alias.py for temporary compatibility${NC}"