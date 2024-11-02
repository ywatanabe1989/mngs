#!/bin/bash
# Time-stamp: "2024-11-03 01:57:46 (ywatanabe)"
# File: ./mngs_repo/docs/generate_test_structure.sh

LOG_FILE="${0%.sh}.log"

usage() {
    echo "Usage: $0 [-f|--force] [-h|--help]"
    echo
    echo "Options:"
    echo " -f, --force  Force overwrite existing test files"
    echo " -h, --help   Display this help message"
    echo
    echo "Example:"
    echo " $0"
    echo " $0 -f"
    exit 1
}

# Parse arguments
FORCE=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -f|--force) FORCE=true ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
    shift
done

SRC="./src/mngs/"
TGT="./tests/"

echo "This will generate test structure. Continue? [y/N]"
read -r response
if [[ ! "$response" =~ ^[yY]$ ]]; then
    echo "Operation cancelled."
    exit 0
fi

{
    echo "Starting test structure generation..."
    echo "Force mode: $FORCE"

    # Create directory structure
    find "$SRC" -type d \
         ! -path "*/\.*" \
         ! -path "*/deprecated*" \
         ! -path "*/archive*" \
         ! -path "*/backup*" \
         ! -path "*/tmp*" \
         ! -path "*/temp*" \
         ! -path "*/__pycache__*" \
         | while read -r dir; do
        test_dir="${dir/src/tests}"
        mkdir -p "$test_dir"
        echo "Created directory: $test_dir"
    done

    # Create test files for each Python file
    find "$SRC" -name "*.py" \
         ! -path "*/\.*" \
         ! -path "*/deprecated*" \
         ! -path "*/archive*" \
         ! -path "*/backup*" \
         ! -path "*/tmp*" \
         ! -path "*/temp*" \
         ! -path "*/__pycache__*" \
         | while read -r src_file; do
        test_file="${src_file/src/tests}"
        test_file="$(dirname "$test_file")/test_$(basename "${test_file%.py}").py"
        base_name=$(basename "$src_file")

        if [ "$base_name" != "__init__.py" ] && [ "$base_name" != "__main__.py" ]; then
            if [[ ! -f "$test_file" ]] || [ "$FORCE" = true ]; then
                cat > "$test_file" << EOF
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pytest
from mngs.${src_file#$SRC} import *

def test_placeholder():
    pass
EOF
                echo "Created test file: $test_file"
            fi
        fi
    done

    echo "Test structure generation completed."
} 2>&1 | tee "$LOG_FILE"

# EOF
