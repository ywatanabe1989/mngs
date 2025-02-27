#!/bin/bash
# -*- coding: utf-8 -*-
# Timestamp: "2025-02-27 22:01:01 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_dev/docs/generate_test_structure.sh

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_PATH="$0.log"
touch "$LOG_PATH"

SRC_DIR="$THIS_DIR/../src/mngs"
TESTS_DIR="$THIS_DIR/../tests/mngs"

# Ensure absolute paths
SRC_DIR=$(realpath "$SRC_DIR")
TESTS_DIR=$(realpath "$TESTS_DIR")

# Function definitions
usage() {
    echo "Usage: $0 [-f|--force] [-u|--update-source] [-h|--help]"
    echo
    echo "Options:"
    echo " -f, --force          Force overwrite existing test files"
    echo " -u, --update-source  Update only source code section"
    echo " -h, --help           Display this help message"
    echo
    echo "Example:"
    echo " $0"
    echo " $0 -f"
    echo " $0 -u"
    exit 1
}

parse_arguments() {
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            -f|--force) FORCE=true ;;
            -u|--update-source) UPDATE_SOURCE=true ;;
            -h|--help) usage ;;
            *) echo "Unknown parameter: $1"; usage ;;
        esac
        shift
    done
}

confirm_operation() {
    if [ "$UPDATE_SOURCE" = false ]; then
        echo "This will generate test structure. Continue? [y/N]"
        read -r response
        [[ ! "$response" =~ ^[yY]$ ]] && echo "Operation cancelled." && exit 0
    fi
}

move_obsolete_files() {
    if [ "$UPDATE_SOURCE" = false ]; then
        OLD_DIR="$TESTS_DIR/.old/$(date +%Y%m%d_%H%M%S)"
        [ -d "$TESTS_DIR" ] && mkdir -p "$OLD_DIR"

        find "$TESTS_DIR" -name "test_*.py" | while read -r test_file; do
            # Skip if test_file is empty
            [ -z "$test_file" ] && continue

            # Skip files in ./tests/custom
            [[ "$test_file" =~ ^./tests/custom ]] && continue

            # Rest of the function remains the same...
            src_file="${test_file#$TESTS_DIR}"
            src_file="${src_file#test_}"
            src_file="$SRC_DIR${src_file}"

            if [ ! -f "$src_file" ]; then
                target_dir="$OLD_DIR/$(dirname "${test_file#$TESTS_DIR}")"
                mkdir -p "$target_dir"
                mv "$test_file" "$target_dir/"
                # echo "Moved obsolete file: $test_file -> $target_dir/"
            fi
        done
    fi
}

create_test_directories() {
    if [ "$UPDATE_SOURCE" = false ]; then
        [ ! -d "$SRC_DIR" ] && echo "Source directory not found: $SRC_DIR" && exit 1

        find "$SRC_DIR" -type d \
             ! -path "*/\.*" \
             ! -path "*/deprecated*" \
             ! -path "*/archive*" \
             ! -path "*/backup*" \
             ! -path "*/tmp*" \
             ! -path "*/temp*" \
             ! -path "*/RUNNING/*" \
             ! -path "*/FINISHED/*" \
             ! -path "*/__pycache__*" \
            | while read -r dir; do
            test_dir="${dir/src/tests}"
            mkdir -p "$test_dir"
            if [[ ! -f "$test_dir/__init__.py" ]] || [ "$FORCE" = true ]; then
                echo "#!/usr/bin/env python3" > "$test_dir/__init__.py"
                echo "# -*- coding: utf-8 -*-" >> "$test_dir/__init__.py"
                # echo "Created __init__.py: $test_dir/__init__.py"
            fi
            # echo "Created directory: $test_dir"
        done
    fi
}

correct_permissions() {
    find "$SRC_DIR" -type f -name "*.py" \
         ! -path "*/\.*" \
         ! -path "*/deprecated*" \
         ! -path "*/archive*" \
         ! -path "*/backup*" \
         ! -path "*/tmp*" \
         ! -path "*/temp*" \
         ! -path "*/RUNNING/*" \
         ! -path "*/FINISHED/*" \
         ! -path "*/__pycache__*" \
        | while read -r script_file; do
        chmod +x $script_file
        done
}

generate_test_template() {
    local src_file=$1
    local test_file=$2

    # Check if source file exists
    [ ! -f "$src_file" ] && echo "Source file not found: $src_file" && return 1

    # Create test file directory if it doesn't exist
    mkdir -p "$(dirname "$test_file")"

    # Convert path from slash to dot notation
    local import_path=${src_file#$SRC_DIR}
    import_path=${import_path%.py}
    import_path=${import_path//\//.}

    cat > "$test_file" << EOF
# src from here --------------------------------------------------------------------------------
$(sed 's/^/# /' "$src_file")

# test from here --------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from ...src.mngs.${import_path} import *

class Test_MainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        pass

    def teardown_method(self):
        # Clean up after tests
        pass

    def test_basic_functionality(self):
        # Basic test case
        pass

    def test_edge_cases(self):
        # Edge case testing
        pass

    def test_error_handling(self):
        # Error handling testing
        pass
EOF
}

create_update_test_files() {
    [ ! -d "$SRC_DIR" ] && echo "Source directory not found: $SRC_DIR" && exit 1

    find "$SRC_DIR" -name "*.py" \
         ! -path "*/\.*" \
         ! -path "*/deprecated*" \
         ! -path "*/archive*" \
         ! -path "*/backup*" \
         ! -path "*/tmp*" \
         ! -path "*/temp*" \
         ! -path "*/RUNNING/*" \
         ! -path "*/FINISHED/*" \
         ! -path "*/__pycache__*" \
        | while read -r src_file; do
        base_name=$(basename "$src_file")

        [[ "$base_name" = "__init__.py" || "$base_name" = "__main__.py" ]] && continue

        test_file="${src_file/src/tests}"
        test_file="$(dirname "$test_file")/test_$(basename "${test_file%.py}").py"

        # Create symlink
        rel_path=$(realpath --relative-to="$(dirname "$test_file")" "$src_file")
        link_file="$(dirname "$test_file")/$(basename "${src_file%.py}")_source.py"
        [ -L "$link_file" ] && rm "$link_file"
        ln -sf "$rel_path" "$link_file"

        # echo "Created symlink: $link_file -> $rel_path"

        if [ "$UPDATE_SOURCE" = true ] && [ -f "$test_file" ]; then
            generate_test_template "$src_file" "$test_file"
            # echo "Updated source in: $test_file"
        elif [[ ! -f "$test_file" ]] || [ "$FORCE" = true ]; then
            generate_test_template "$src_file" "$test_file"
            # echo "Created test file: $test_file"
        fi
    done
}

main() {
    FORCE=false
    UPDATE_SOURCE=false

    parse_arguments "$@"
    confirm_operation
    correct_permissions

    {
        echo "Starting test structure generation..."
        echo "Force mode: $FORCE"
        echo "Update source only: $UPDATE_SOURCE"

        move_obsolete_files
        create_test_directories
        create_update_test_files

        echo "Test structure generation completed."

        tree ./tests
    } 2>&1 | tee "$LOG_PATH"
}

main "$@"

# EOF