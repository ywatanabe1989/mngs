# --------------------------------------------------------------------------------
# Start of Source code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/decorators/__init__.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-02-27 11:16:30 (ywatanabe)"
# # File: ./src/mngs/decorators/__init__.py
# 
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/__init__.py"
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2025-02-27 11:16:30 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/decorators/__init__.py
# 
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/__init__.py"
# 
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 13:05:15 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/decorators/__init__.py
# 
# # import os
# # import importlib
# # import inspect
# 
# # # Get the current directory
# # current_dir = os.path.dirname(__file__)
# 
# # # Iterate through all Python files in the current directory
# # for filename in os.listdir(current_dir):
# #     if filename.endswith(".py") and not filename.startswith("__"):
# #         module_name = filename[:-3]  # Remove .py extension
# #         module = importlib.import_module(f".{module_name}", package=__name__)
# 
# #         # Import only functions and classes from the module
# #         for name, obj in inspect.getmembers(module):
# #             if inspect.isfunction(obj) or inspect.isclass(obj):
# #                 if not name.startswith("_"):
# #                     globals()[name] = obj
# 
# # # Clean up temporary variables
# # del (
# #     os,
# #     importlib,
# #     inspect,
# #     current_dir,
# #     filename,
# #     module_name,
# #     module,
# #     name,
# #     obj,
# # )
# 
# from ._cache_disk import *
# from ._cache_mem import *
# from ._converters import *
# from ._DataTypeDecorators import *
# from ._deprecated import *
# from ._not_implemented import *
# from ._numpy_fn import *
# from ._pandas_fn import *
# from ._preserve_doc import *
# from ._preserve_docstring import *
# from ._timeout import *
# from ._torch_fn import *
# from ._wrap import *
# 
# # EOF#
# --------------------------------------------------------------------------------
# End of Source code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/decorators/__init__.py
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import pytest
    pytest.main([os.path.abspath(__file__)])
