#!/usr/bin/env python3

import numpy as np

from ._PARAMS import PARAMS


def to_rgb(c):
    return PARAMS["RGB"][c]


def to_rgba(c, alpha=0.5):
    rgba = rgb2rgba(PARAMS["RGB"][c])
    rgba[-1] = alpha
    return rgba


def to_hex(c):
    return PARAMS["HEX"][c]


def rgb2rgba(rgb, alpha=0, round=2):
    rgb = np.array(rgb).astype(float)
    rgb /= 255
    return [*rgb.round(round), alpha]


def update_alpha(rgba, alpha):
    rgba_list = list(rgba)
    rgba_list[-1] = alpha
    return rgba_list


def rgba2hex(rgba):
    return "#{:02x}{:02x}{:02x}{:02x}".format(
        int(rgba[0]), int(rgba[1]), int(rgba[2]), int(rgba[3] * 255)
    )


def cycle_color(i_color):
    COLORS_10_STR = list(PARAMS["RGB"].keys())
    n_colors = len(COLORS_10_STR)
    return COLORS_10_STR[i_color % n_colors]


if __name__ == "__main__":
    c = "blue"
    print(to_rgb(c))
    print(to_rgba(c))
    print(to_hex(c))
    print(cycle_color(1))
