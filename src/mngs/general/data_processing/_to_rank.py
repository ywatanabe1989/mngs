#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-08-29 22:10:06 (ywatanabe)"
# /home/ywatanabe/proj/mngs_repo/src/mngs/general/data_processing/_to_rank.py

import torch
# from ..decorators._converters import
from mngs.gen import torch_fn


@torch_fn
def to_rank(tensor, method="average"):
    sorted_tensor, indices = torch.sort(tensor)
    ranks = torch.empty_like(tensor)
    ranks[indices] = (
        torch.arange(len(tensor), dtype=tensor.dtype, device=tensor.device) + 1
    )

    if method == "average":
        ranks = ranks.float()
        ties = torch.nonzero(sorted_tensor[1:] == sorted_tensor[:-1])
        for i in range(len(ties)):
            start = ties[i]
            end = start + 1
            while (
                end < len(sorted_tensor)
                and sorted_tensor[end] == sorted_tensor[start]
            ):
                end += 1
            ranks[indices[start:end]] = ranks[indices[start:end]].mean()

    return ranks
