#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-10 10:07:22 (ywatanabe)"

from mngs.gen import deprecated


@deprecated("Use plot_with_ci(0 instaed.")
def fill_between(ax, xx, mean, std, **kwargs):
    ax.plot(xx, mean, **kwargs)
    ax.fill_between(xx, mean - std, mean + std, **kwargs)
    return ax


def plot_with_ci(ax, xx, mean, std, **kwargs):
    return fill_between(ax, xx, mean, std, **kwargs)
