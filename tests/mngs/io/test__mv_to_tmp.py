# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_mv_to_tmp.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 21:25:50 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/io/_mv_to_tmp.py
# 
# from shutil import move
# 
# def _mv_to_tmp(fpath, L=2):
#     try:
#         tgt_fname = "-".join(fpath.split("/")[-L:])
#         tgt_fpath = "/tmp/{}".format(tgt_fname)
#         move(fpath, tgt_fpath)
#         print("Moved to: {}".format(tgt_fpath))
#     except:
#         pass
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_mv_to_tmp.py
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import pytest
    pytest.main([os.path.abspath(__file__)])
