#!/bin/bash
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-28 00:19:20 (ywatanabe)"
# File: ./run_tests.sh

THIS_DIR="$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)"
LOG_PATH="$THIS_DIR/.$(basename $0).log"
touch "$LOG_PATH" >/dev/null 2>&1


# find . -name "__pycache__" -type d -exec rm -rf {} +
# find . -name "*.pyc" -delete

pytest | tee "$LOG_PATH"

# EOF