#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-06-04 07:23:52 (ywatanabe)"

from mngs.gen import deprecated


@deprecated("Use plot_with_ci(0 instaed.")
def fill_between(ax, xx, mean, std, label=None, alpha=0.1):
    ax.plot(xx, mean, label=label, alpha=alpha)
    ax.fill_between(xx, mean - std, mean + std, alpha=alpha)
    return ax


def plot_with_ci(ax, xx, mean, std, label=None, alpha=0.1):
    return fill_between(ax, xx, mean, std, label=None, alpha=0.1)
