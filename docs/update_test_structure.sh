#!/bin/bash
# -*- coding: utf-8 -*-
# Timestamp: "2025-02-28 00:05:19 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_dev/docs/update_test_structure.sh

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_PATH="$0.log"
touch "$LOG_PATH"

SRC_DIR=$(realpath "$THIS_DIR/../src/mngs")
TESTS_DIR=$(realpath "$THIS_DIR/../tests/mngs")

# Function definitions
usage() {
    echo "Usage: $0 [-h|--help]"
    echo
    echo "Update or create test files under './tests' directory."
    echo "Test files without matching source files are moved to a '.old' directory."
    echo
    echo "Use './tests/custom' for customized tests"
    echo
    echo "Options:"
    echo " -h, --help           Display this help message."
    echo
    echo "Example:"
    echo " $0                   # Generate or update test files."
    exit 1
}

parse_arguments() {
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            -h|--help) usage ;;
            *) echo "Unknown parameter: $1"; usage ;;
        esac
        shift
    done
}

construct_exclude_patterns() {
    # Constructs the exclude patterns for 'find' command
    local EXCLUDE_PATHS=(
        "*/\.*"
        "*/deprecated*"
        "*/archive*"
        "*/backup*"
        "*/tmp*"
        "*/temp*"
        "*/RUNNING/*"
        "*/FINISHED/*"
        "*/__pycache__*"
        "*/__init__.py*"
    )

    # Build the find parameters for exclusion globally
    FIND_EXCLUDES=()
    for path in "${EXCLUDE_PATHS[@]}"; do
        FIND_EXCLUDES+=(-not -path "$path")
    done
}

find_files() {
    # Usage: find_files <search_path> <type> <name_pattern>
    # Example: find_files "$SRC_DIR" f "*.py"
    local search_path="$1"
    local type="$2"
    local name_pattern="$3"

    # Call the function to populate FIND_EXCLUDES
    construct_exclude_patterns

    find "$search_path" -type "$type" -name "$name_pattern" "${FIND_EXCLUDES[@]}"
}

move_obsolete_files_to_old() {
    OLD_DIR="$TESTS_DIR/.old/$(date +%Y%m%d_%H%M%S)"
    [ -d "$TESTS_DIR" ] && mkdir -p "$OLD_DIR"

    find "$TESTS_DIR" -name "test_*.py" | while read -r test_file; do
        # Skip if test_file is empty
        [ -z "$test_file" ] && continue

        # Skip files in ./tests/custom
        [[ "$test_file" =~ ^${TESTS_DIR}/custom ]] && continue

        # Determine corresponding source file
        relative_path="${test_file#$TESTS_DIR/}"
        src_relative_path="${relative_path#test_}"
        src_file="$SRC_DIR/$src_relative_path"

        if [ ! -f "$src_file" ]; then
            target_dir="$OLD_DIR/$(dirname "$relative_path")"
            mkdir -p "$target_dir"
            mv "$test_file" "$target_dir/"
            # echo "Moved obsolete file: $test_file -> $target_dir/"
        fi
    done
}

create_test_directories() {
    [ ! -d "$SRC_DIR" ] && echo "Source directory not found: $SRC_DIR" && exit 1

    construct_exclude_patterns
    find "$SRC_DIR" -type d "${FIND_EXCLUDES[@]}" | while read -r dir; do
        test_dir="${dir/$SRC_DIR/$TESTS_DIR}"
        mkdir -p "$test_dir"
        # echo "Created directory: $test_dir"
    done
}

correct_permissions() {
    construct_exclude_patterns
    find "$SRC_DIR" -type f -name "*.py" "${FIND_EXCLUDES[@]}" | while read -r script_file; do
        chmod +x "$script_file"
    done
}

# New function to upsert the test header
upsert_test_header() {
    local test_file=$1
    cat << EOF
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import pytest
import numpy as np

EOF
}

# New function to upsert the source code as comments
upsert_source_code_as_comment() {
    local src_file=$1
    echo "# Source code from: $src_file"
    echo "# --------------------------------------------------------------------------------"
    sed 's/^/# /' "$src_file"
    echo
}

# New function to upsert the test code
upsert_test_code() {
    local src_file=$1
    local test_file=$2

    # Convert path from slash to dot notation
    local import_path=${src_file#$SRC_DIR}
    import_path=${import_path%.py}
    import_path=${import_path//\//.}

    cat << EOF
# Add source code to the top of Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from mngs${import_path} import *

class TestMainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        pass

    def teardown_method(self):
        # Clean up after tests
        pass

    def test_basic_functionality(self):
        # Basic test case
        raise NotImplementedError("Test not yet implemented")

    def test_edge_cases(self):
        # Edge case testing
        raise NotImplementedError("Test not yet implemented")

    def test_error_handling(self):
        # Error handling testing
        raise NotImplementedError("Test not yet implemented")

if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
EOF
}

generate_test_template() {
    local src_file=$1
    local test_file=$2

    # Check if source file exists
    if [ ! -f "$src_file" ]; then
        echo "Source file not found: $src_file"
        return 1
    fi

    # Create test file directory if it doesn't exist
    mkdir -p "$(dirname "$test_file")"

    {
        upsert_source_code_as_comment "$src_file"
        upsert_test_header "$test_file"
        upsert_test_code "$src_file" "$test_file"
    } > "$test_file"
}

create_update_test_files() {
    [ ! -d "$SRC_DIR" ] && echo "Source directory not found: $SRC_DIR" && exit 1

    construct_exclude_patterns
    find "$SRC_DIR" -type f -name "*.py" "${FIND_EXCLUDES[@]}" | while read -r src_file; do
        base_name=$(basename "$src_file")

        [[ "$base_name" = "__init__.py" || "$base_name" = "__main__.py" ]] && continue

        test_file="${src_file/$SRC_DIR/$TESTS_DIR}"
        test_file="$(dirname "$test_file")/test_${base_name%.py}.py"

        # Create symlink
        rel_path=$(realpath --relative-to "$(dirname "$test_file")" "$src_file")
        link_file="$(dirname "$test_file")/${base_name%.py}_source.py"
        [ -L "$link_file" ] && rm "$link_file"
        ln -sf "$rel_path" "$link_file"

        # Update or create test files
        generate_test_template "$src_file" "$test_file"
    done
}

main() {
    parse_arguments "$@"
    correct_permissions

    {
        move_obsolete_files_to_old
        create_test_directories
        create_update_test_files

        echo "Test structure generation completed."

        tree "$TESTS_DIR"
    } 2>&1 | tee "$LOG_PATH"
}

main "$@"

# EOF