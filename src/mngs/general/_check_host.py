#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-13 18:04:25 (ywatanabe)"
# /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/general/_check_host.py

def check_host(keyword):
    import mngs
    return keyword in mngs.sh('echo $(hostname)', verbose=False)

is_host = check_host


def verify_host(keyword):
    import mngs
    import sys

    if is_host(keyword):
        print(f"Host verification successed for keyword: {keyword}")
        return
    else:
        print(f"Host verification failed for keyword: {keyword}")
        sys.exit(1)

if __name__ == '__main__':
    # check_host("ywata")
    verify_host("titan")
    verify_host("ywata")
    verify_host("crest")
