# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-07 17:17:06 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/dev/_reload.py
# 
# 
# import importlib
# import sys
# import threading
# import time
# from typing import Any, Optional
# 
# _reload_thread: Optional[threading.Thread] = None
# _running: bool = False
# 
# 
# def reload() -> Any:  # Changed return type hint to Any
#     """Reloads mngs package and its submodules."""
#     import mngs
# 
#     mngs_modules = [mod for mod in sys.modules if mod.startswith("mngs")]
#     for module in mngs_modules:
#         try:
#             importlib.reload(sys.modules[module])
#         except Exception:
#             pass
#     return importlib.reload(mngs)
# 
# 
# def reload_auto(interval: int = 10) -> None:
#     """Start auto-reload in background thread."""
#     global _reload_thread, _running
# 
#     if _reload_thread and _reload_thread.is_alive():
#         return
# 
#     _running = True
#     _reload_thread = threading.Thread(
#         target=_auto_reload_loop, args=(interval,), daemon=True
#     )
#     _reload_thread.start()
# 
# 
# def reload_stop() -> None:
#     """Stop auto-reload."""
#     global _running
#     _running = False
# 
# 
# def _auto_reload_loop(interval: int) -> None:
#     while _running:
#         try:
#             reload()
#         except Exception as e:
#             print(f"Reload failed: {e}")
#         time.sleep(interval)
# 
# 
# # EOF

# test from here --------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from mngs..dev._reload import *

class Test_MainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        pass

    def teardown_method(self):
        # Clean up after tests
        pass

    def test_basic_functionality(self):
        # Basic test case
        pass

    def test_edge_cases(self):
        # Edge case testing
        pass

    def test_error_handling(self):
        # Error handling testing
        pass
