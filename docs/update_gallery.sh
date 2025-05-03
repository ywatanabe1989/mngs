#!/bin/bash
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-03 16:53:13 (ywatanabe)"
# File: ./docs/update_plot_gallery.sh

THIS_DIR="$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)"
LOG_PATH="$THIS_DIR/.$(basename $0).log"
touch "$LOG_PATH" >/dev/null 2>&1


GALLERY_PATH=./src/mngs/plt/gallery.md
# Find all test image files and process them
for f in $(find tests/mngs -type f -name "test*.jpg"); do
    echo "Processing: $f" | tee -a "$LOG_PATH"

    # Extract filename without path for display purposes
    filename=$(basename "$f")

    # Add entry to the gallery markdown file
    echo "## $filename" >> $GALLERY_PATH
    echo "![Test Image]($f)" >> $GALLERY_PATH
    echo "" >> $GALLERY_PATH
done

echo "Gallery generation complete. Results saved to $GALLERY_PATH" | tee -a "$LOG_PATH"

# EOF