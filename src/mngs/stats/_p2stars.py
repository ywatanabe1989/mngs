#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-10 13:47:03 (ywatanabe)"


def p2stars(pvalue, ns=False):
    if pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    else:
        return "n.s." if ns else ""
