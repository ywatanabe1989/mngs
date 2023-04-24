#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-02-16 08:13:57 (ywatanabe)"

import scipy

def gaussian_filter1d(xx, radius):
    # radius = round(truncate * sigma)
    sigma = 1
    truncate = radius / sigma
    return scipy.ndimage.gaussian_filter1d(xx, sigma, truncate=truncate)
    
    
    
