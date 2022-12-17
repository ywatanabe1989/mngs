#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-10-07 11:28:33 (ywatanabe)"

import numpy as np


def take_random_references(sig_2D, random_state=42):
    n_chs = len(sig_2D)
    rs = np.random.RandomState(random_state)
    ref_sig_2D = sig_2D[rs.permutation(np.arange(n_chs))]
    return sig_2D - ref_sig_2D
