#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-02 14:27:02 (ywatanabe)"
# File: ./mngs_repo/src/mngs/utils/_gen_ID.py

def gen_ID(time_format="%YY-%mM-%dD-%Hh%Mm%Ss", N=8):
    import random
    import string
    from datetime import datetime

    now = datetime.now()
    # now_str = now.strftime("%Y-%m-%d-%H-%M")
    now_str = now.strftime(time_format)

    # today_str = now.strftime("%Y-%m%d")
    randlst = [
        random.choice(string.ascii_letters + string.digits) for i in range(N)
    ]
    rand_str = "".join(randlst)
    return now_str + "_" + rand_str


# EOF
