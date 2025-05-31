<!-- ---
!-- Timestamp: 2025-05-31 07:42:09
!-- Author: ywatanabe
!-- File: /ssh:ywatanabe@sp:/home/ywatanabe/.dotfiles/.claude/to_claude/guidelines/python/IMPORTANT-MNGS-02-file-template.md
!-- --- -->

## Script Template

MNGS PYTHON SCRIPT MUST STRICTLY FOLLOW THIS STANDARD FORMAT:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-03 10:33:13 (ywatanabe)"
# File: ./relative/path/from/project/root/script_name.py

__FILE__ = "./relative/path/from/project/root/script_name.py"

"""
Functionalities:
  - Does XYZ
  - Saves XYZ

Dependencies:
  - scripts: /path/to/script1, /path/to/script2
  - packages: package1, package2

IO:
  - input-files: /path/to/input/file.xxx
  - output-files: /path/to/output/file.xxx
"""

"""Imports"""
import os
import sys
import argparse

"""Parameters"""
# from mngs.io import load_configs
# CONFIG = load_configs()

"""Functions & Classes"""
def main(args):
    # Main functionality goes here
    pass

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    import mngs
    script_mode = mngs.gen.is_script()
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    mngs.str.printc(args, c='yellow')
    return args

def run_main() -> None:
    """Initialize mngs framework, run main function, and cleanup."""
    global CONFIG, CC, sys, plt

    import sys
    import matplotlib.pyplot as plt
    import mngs

    args = parse_args()

    # Start mngs framework
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        args=args,
        file=__FILE__,
        verbose=False,
        agg=True,
    )

    # Main
    exit_status = main(args)

    # Close the mngs framework
    mngs.gen.close(
        CONFIG,
        verbose=False,
        notify=False,
        message="",
        exit_status=exit_status,
    )

if __name__ == '__main__':
    run_main()

# EOF
```

**⚠️ DO NOT MODIFY THE `run_main()` FUNCTION**  
This handles stdout/stderr direction, logging, configuration, and more

## Your Understanding Check
Did you understand the guideline? If yes, please say:
`CLAUDE UNDERSTOOD: <THIS FILE PATH HERE>`

<!-- EOF -->