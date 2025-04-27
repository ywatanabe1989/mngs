#!/bin/bash
# -*- coding: utf-8 -*-
# Timestamp: "2025-02-27 22:17:47 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_dev/docs/memo_test.sh

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_PATH="$0.log"
touch "$LOG_PATH"

generate_test_structure.sh -u

# EOF