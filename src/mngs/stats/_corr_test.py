#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-03-14 13:52:41 (ywatanabe)"

from bisect import bisect_right
import numpy as np
import mngs
from copy import deepcopy

def corr_test(d1, d2, only_significant=False):
    d1 = deepcopy(d1)
    d2 = deepcopy(d2)
    nan_indi = np.isnan(d1) + np.isnan(d2)
    d1 = d1[~nan_indi]
    d2 = d2[~nan_indi]    
    mngs.gen.fix_seeds(42, np=np, show=False)
    corr_obs = np.corrcoef(d1, d2)[0,1]
    corrs_shuffled = [np.corrcoef(d1, np.random.permutation(d2))[0,1]
                     for _ in range(1000)
                     ]
    rank = bisect_right(corrs_shuffled, corr_obs)

    mid_rank = len(corrs_shuffled) // 2

    if rank <= mid_rank:
        rank /= 2
    if mid_rank < rank:
        rank = (len(corrs_shuffled) - rank) // 2            
    pval = rank / len(corrs_shuffled)

    mark = mngs.stats.to_asterisks(pval)        

    if not only_significant:
        print(f"Corr. = {round(corr_obs, 2)}; p-val = {round(pval, 3)}", mark)        
    if only_significant and (pval < 0.05):
        print(round(corr_obs, 2), round(pval, 3), mark)                

    return round(pval, 3), round(corr_obs, 2), np.array(corrs_shuffled)

if __name__ == "__main__":
    x = np.array([3, 4, 4, 5, 7, 8, 10, 12, 13, 15])
    y = np.array([2, 4, 4, 5, 4, 7, 8, 19, 14, 10])
    pval, corr_obs, corrs_shuffled = corr_test(x,y)
    
