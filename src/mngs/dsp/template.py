#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-10 15:57:54 (ywatanabe)"



# FUnctions

if __name__ == "__main__":
    import sys

    import matplotlib.pyplot as plt
    
    import torch

    # Start
    CONFIG, sys.stdout, sys.stderr, plt, cc = mngs.gen.start(sys, plt)

    # Close
    mngs.gen.close(CONFIG)

    """
    /home/ywatanabe/proj/mngs/src/mngs/dsp/template.py
    """

    # EOF
