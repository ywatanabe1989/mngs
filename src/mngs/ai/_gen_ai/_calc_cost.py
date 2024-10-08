#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-29 15:24:20 (ywatanabe)"
# /home/ywatanabe/proj/mngs_repo/src/mngs/ai/_gen_ai/_PRICING.py


from .PARAMS import MODELS


def calc_cost(model, input_tokens, output_tokens):
    indi = MODELS["name"] == model
    costs = MODELS[["input_cost", "output_cost"]][indi]
    cost = (
        input_tokens * costs["input_cost"]
        + output_tokens * costs["output_cost"]
    ) / 1_000_000
    return cost.iloc[0]
