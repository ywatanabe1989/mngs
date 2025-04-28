# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import pytest
    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/dev/_reload.py
# --------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/dev/_reload.py
# --------------------------------------------------------------------------------
