#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-10-07 11:05:20 (ywatanabe)"


def common_average(sig_2D):
    return (sig_2D - sig_2D.mean()) / sig_2D.std()
