#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-03-17 11:06:18 (ywatanabe)"

def ax_fill_between(ax, xx, mean, std, label, alpha=1):
    ax.plot(xx, mean, label=label, alpha=alpha)
    ax.fill_between(xx, mean-std, mean+std, alpha=0.1)
    return ax
