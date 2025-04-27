#!/bin/bash
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-28 07:20:39 (ywatanabe)"
# File: ./run_tests.sh

THIS_DIR="$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)"
LOG_PATH="$THIS_DIR/.$(basename $0).log"
touch "$LOG_PATH" >/dev/null 2>&1


# Default settings
DELETE_CACHE=false
UPDATE_TEST_STRUCTURE=false
VERBOSE=false
SPECIFIC_TEST=""
PYTEST_INI_PATH=$THIS_DIR/tests/pytest.ini
N_WORKERS=$(nproc)
if [ $? -ne 0 ]; then
    N_WORKERS=$(sysctl -n hw.ncpu 2>/dev/null || echo 1)
fi
N_WORKERS=$((N_WORKERS * 3 / 4))  # Use 75% of cores


usage() {
    echo "Usage: $0 [options] [test_path]"
    echo
    echo "Options:"
    echo "  -c, --cache        Delete Python cache files (default: $DELETE_CACHE)"
    echo "  -u, --update       Update test structure (default: $UPDATE_TEST_STRUCTURE)"
    echo "  -n, --n_workers    Number of workers (default: $N_WORKERS, auto-parallel if >1)"
    echo "  -v, --verbose      Run tests in verbose mode (default: $VERBOSE)"
    echo "  -h, --help         Display this help message"
    echo
    echo "Arguments:"
    echo "  test_path          Optional path to specific test file or directory"
    echo
    echo "Example:"
    echo "  $0 -c -u           Clean cache and update test structure"
    echo "  $0 -n 4            Run tests in parallel with 4 workers"
    echo "  $0 tests/mngs/core Run only tests in core module"
    exit 1
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--cache)
                DELETE_CACHE=true
                shift
                ;;
            -u|--update)
                UPDATE_TEST_STRUCTURE=true
                shift
                ;;
            -n|--n_workers)
                N_WORKERS="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                usage
                ;;
            *)
                if [[ -e "$1" ]]; then
                    SPECIFIC_TEST="$1"
                    shift
                else
                    echo "Unknown option or file not found: $1"
                    usage
                fi
                ;;
        esac
    done
}

clean_cache() {
    echo "Cleaning Python cache..."
    find . -name "__pycache__" -type d -exec rm -rf {} \; 2>/dev/null || true
    find . -name "*.pyc" -delete
}

update_test_structure() {
    echo "Updating test structure..."
    "$THIS_DIR/tests/update_test_structure.sh"
}

run_tests() {
    PYTEST_ARGS="-c $PYTEST_INI_PATH"

    if [[ $N_WORKERS -gt 1 ]]; then
        echo "Running in parallel mode with $N_WORKERS workers"
        PYTEST_ARGS="$PYTEST_ARGS -n $N_WORKERS"
    fi

    if [[ $VERBOSE == true ]]; then
        PYTEST_ARGS="$PYTEST_ARGS -v"
    fi

    if [[ -n "$SPECIFIC_TEST" ]]; then
        echo "Running specific test: $SPECIFIC_TEST"
        PYTEST_ARGS="$PYTEST_ARGS $SPECIFIC_TEST"
    fi

    echo "Running pytest..."
    pytest $PYTEST_ARGS | tee "$LOG_PATH"

    local exit_code=${PIPESTATUS[0]}
    echo "Test execution completed with exit code $exit_code. Log saved to $LOG_PATH"
    return $exit_code
}

main() {
    parse_args "$@"

    if [[ $DELETE_CACHE == true ]]; then
        clean_cache
    fi

    if [[ $UPDATE_TEST_STRUCTURE == true ]]; then
        update_test_structure
    fi

    run_tests
    return $?
}

# Execute main function with all arguments
main "$@"

# EOF