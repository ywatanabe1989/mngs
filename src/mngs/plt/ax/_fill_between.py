#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-07-10 19:26:42 (ywatanabe)"

def fill_between(ax, xx, mean, std, label, alpha=.1):
    ax.plot(xx, mean, label=label, alpha=alpha)
    ax.fill_between(xx, mean-std, mean+std, alpha=alpha)
    return ax
