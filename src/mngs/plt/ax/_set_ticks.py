#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-07 15:39:32 (ywatanabe)"

import numpy as np


def set_ticks(ax, xticks=None, yticks=None):

    if xticks is not None:
        ax.set_xticks(np.arange(0, len(xticks)))
        ax.set_xticklabels(xticks)

    if yticks is not None:
        ax.set_yticks(np.arange(0, len(yticks)))
        ax.set_yticklabels(yticks)

    return ax
