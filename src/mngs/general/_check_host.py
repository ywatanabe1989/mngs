#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-12 12:14:40 (ywatanabe)"
# /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/general/_check_host.py

def check_host(keyword):
    import mngs
    return keyword in mngs.sh('echo $(hostname)', verbose=False)


if __name__ == '__main__':
    check_host("ywata")
