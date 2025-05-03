# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/resource/limit_RAM.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2021-09-20 21:02:04 (ywatanabe)"
# 
# import resource
# import mngs
# 
# 
# def limit_RAM(RAM_factor):
#     soft, hard = resource.getrlimit(resource.RLIMIT_AS)
#     max_val = min(RAM_factor * get_RAM() * 1024, get_RAM() * 1024)
#     resource.setrlimit(resource.RLIMIT_AS, (max_val, hard))
#     print(f"\nFree RAM was limited to {mngs.gen.fmt_size(max_val)}")
# 
# 
# def get_RAM():
#     with open("/proc/meminfo", "r") as mem:
#         free_memory = 0
#         for i in mem:
#             sline = i.split()
#             if str(sline[0]) in ("MemFree:", "Buffers:", "Cached:"):
#                 free_memory += int(sline[1])
#     return free_memory
# 
# 
# if __name__ == "__main__":
#     get_RAM()
#     limit_RAM(0.1)

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/resource/limit_RAM.py
# --------------------------------------------------------------------------------
