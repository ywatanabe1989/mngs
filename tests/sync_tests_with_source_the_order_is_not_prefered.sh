#!/bin/bash
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-28 15:36:02 (ywatanabe)"
# File: ./tests/sync_tests_with_source.sh

THIS_DIR="$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)"
LOG_PATH="$THIS_DIR/.$(basename $0).log"
touch "$LOG_PATH" >/dev/null 2>&1


THIS_DIR="./tests"

ORIG_DIR="$(pwd)"
ROOT_DIR="$(realpath $THIS_DIR/..)"

cmd="cd $ROOT_DIR" && echo "$cmd" && eval "$cmd"

SRC_DIR="$(realpath "${THIS_DIR}/../src/mngs")"
TESTS_DIR="$(realpath "${THIS_DIR}/../tests/mngs")"

########################################
# Test Structure
########################################
prepare_tests_structure_as_source() {
    [ ! -d "$SRC_DIR" ] && echo "Source directory not found: $SRC_DIR" && exit 1
    construct_blacklist_patterns
    find "$SRC_DIR" -type d "${FIND_EXCLUDES[@]}" | while read -r dir; do
        tests_dir="${dir/$SRC_DIR/$TESTS_DIR}"
        mkdir -p "$tests_dir"
    done
}


# prepare_tests_structure_as_source() {
#     [ ! -d "$SRC_DIR" ] && echo "Source directory not found: $SRC_DIR" && exit 1
#     construct_blacklist_patterns
#     find "$SRC_DIR" -type d "${FIND_EXCLUDES[@]}" | while read -r dir; do
#         # Skip directories that start with a dot (hidden directories)
#         [[ $(basename "$dir") == .* ]] && continue
#         tests_dir="${dir/$SRC_DIR/$TESTS_DIR}"
#         mkdir -p "$tests_dir"
#     done
# }

########################################
# Source as Comment
########################################
get_source_code_start_tag() {
    local src_file=$1
    printf "%s\n" \
        "# --------------------------------------------------------------------------------" \
        "# Start of Source Code from: $src_file" \
        "# --------------------------------------------------------------------------------"
}

get_source_code_end_tag() {
    local src_file=$1
    printf "%s\n" \
        "" \
        "# --------------------------------------------------------------------------------" \
        "# End of Source Code from: $src_file" \
        "# --------------------------------------------------------------------------------"
}

get_source_code_as_comment() {
    local src_file=$1
    get_source_code_start_tag "$src_file"
    sed 's/^/# /' "$src_file"
    get_source_code_end_tag "$src_file"
    echo
}

find_source_code_comment() {
    local test_file=$1

    local start_line_tag
    start_line_tag=$(grep -n '^# Start of Source Code from:' "$test_file" \
                     | cut -d: -f1)

    local start_line
    start_line=$((start_line_tag - 1))

    local end_line_tag
    end_line_tag=$(grep -n '^# End of Source Code from:' "$test_file" \
                   | cut -d: -f1)

    local end_line
    end_line=$((end_line_tag + 1))

    printf "%s %s" "$start_line" "$end_line"
}

replace_or_add_source_code_comment() {
    local test_file=$1
    local src_file=$2
    local new_block=$(get_source_code_as_comment "$src_file")

    if [ ! -f "$test_file" ]; then
        echo "$test_file not found. Creating..."
        mkdir -p "$(dirname "$test_file")"
        echo "$new_block" > "$test_file"
    else
        read start_line end_line < <(find_source_code_comment "$test_file")

        # If no comment block exists yet
        if [ -z "$start_line" ] || [ -z "$end_line" ]; then
            printf '%s\n' "$new_block" > "$test_file.new"
            cat "$test_file" | tee -a "$test_file.new"
            mv "$test_file.new" "$test_file"
        else
            # Using temporary files for portability
            head -n $((start_line-1)) "$test_file" > "$test_file.tmp"
            echo "$new_block" | tee -a "$test_file.tmp"
            tail -n +$((end_line+1)) "$test_file" | tee -a "$test_file.tmp"
            mv "$test_file.tmp" "$test_file"
        fi
    fi
}

########################################
# Pytest Main Guard
########################################
add_pytest_main_guard_if_not_present() {
    local test_file=$1
    # Only add if file exists and doesn't already have the guard
    [ ! -f "$test_file" ] && return
    grep -q '^if __name__ == "__main__":' "$test_file" && return

    # Add the guard at the end
    cat | tee -a "$test_file" << 'EOL'

if __name__ == "__main__":
    import os
    import pytest
    pytest.main([os.path.abspath(__file__)])
EOL
}

########################################
# Finder
########################################
construct_blacklist_patterns() {
   local EXCLUDE_PATHS=(
       "*/.*"
       "*/.*/*"
       "*/deprecated*"
       "*/archive*"
       "*/backup*"
       "*/tmp*"
       "*/temp*"
       "*/RUNNING/*"
       "*/FINISHED/*"
       "*/FINISHED_SUCCESS/*"
       "*/2025Y*"
       "*/2024Y*"
       "*/__pycache__/*"
   )
   FIND_EXCLUDES=()
   PRUNE_ARGS=()
   for path in "${EXCLUDE_PATHS[@]}"; do
       FIND_EXCLUDES+=( -not -path "$path" )
       PRUNE_ARGS+=( -path "$path" -o )
   done
   unset 'PRUNE_ARGS[${#PRUNE_ARGS[@]}-1]'
}

find_files() {
   local search_path=$1
   local type=$2
   local name_pattern=$3
   construct_blacklist_patterns
   find "$search_path" \
       \( "${PRUNE_ARGS[@]}" \) -prune -o -type "$type" -name "$name_pattern" -print
}

########################################
# Clean-upper
########################################
move_stale_test_files_to_old() {

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
            # echo "Moved stale file: $test_file -> $target_dir/"
        fi
    done
}

remove_hidden_test_files_and_dirs() {
    find "$TESTS_DIR" -type f -name ".*" -exec rm -rf {} +
    find "$TESTS_DIR" -type d -name ".*" -exec rm -rf {} +
}

cleanup_unnecessary_test_files() {
    find "$TESTS_DIR" -type d -name "*RUNNING*" -exec rm -rf {} +
    find "$TESTS_DIR" -type d -name "*FINISHED*" -exec rm -rf {} +
    find "$TESTS_DIR" -type d -name "*FINISHED_SUCCESS*" -exec rm -rf {} +
    find "$TESTS_DIR" -type d -name "*2024Y*" -exec rm -rf {} +
    find "$TESTS_DIR" -type d -name "*2025Y*" -exec rm -rf {} +
    find "$TESTS_DIR" -type d -name "*.py_out" -exec rm -rf {} +
}

########################################
# Permission
########################################
chmod_python_source_scripts_as_executable() {
    construct_blacklist_patterns
    find "$SRC_DIR" -type f -name "*.py" "${FIND_EXCLUDES[@]}" | while read -r script_file; do
        chmod +x "$script_file"
    done
}

########################################
# Main
########################################
main() {
    echo "Using SRC_DIR: $SRC_DIR"
    echo "Using TESTS_DIR: $TESTS_DIR"

    remove_hidden_test_files_and_dirs

    prepare_tests_structure_as_source
    chmod_python_source_scripts_as_executable

    # update tests: embed source comments & pytest guard
    find_files "$SRC_DIR" f "*.py" | while read -r src_file; do
        # derive relative path and parts
        rel="${src_file#$SRC_DIR/}"
        rel_dir=$(dirname "$rel")
        src_base=$(basename "$rel")

        # ensure test subdir exists
        tests_dir="$TESTS_DIR/$rel_dir"
        mkdir -p "$tests_dir"

        # build correct test file path
        test_file="$tests_dir/test_$src_base"

        replace_or_add_source_code_comment "$test_file" "$src_file"
        add_pytest_main_guard_if_not_present "$test_file"
    done

    remove_hidden_test_files_and_dirs
    # cleanup_unnecessary_test_files
    # move_stale_test_files_to_old

    tree "$TESTS_DIR" 2>&1 | tee -a "$LOG_PATH"
}

main "$@"

cd $ORIG_DIR

# EOF