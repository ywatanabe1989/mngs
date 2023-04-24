#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-03-03 07:47:14 (ywatanabe)"

def to_asterisks(pvalue):
    if pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"
