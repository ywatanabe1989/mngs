# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_load_modules/_catboost.py
# --------------------------------------------------------------------------------
# # #!/usr/bin/env python3
# # # -*- coding: utf-8 -*-
# # # Time-stamp: "2024-12-12 06:50:19 (ywatanabe)"
# # # File: ./mngs_repo/src/mngs/io/_load_modules/_catboost.py
# 
# # from typing import Union
# 
# # from catboost import CatBoostClassifier, CatBoostRegressor
# 
# 
# # def _load_catboost(
# #     lpath: str, **kwargs
# # ) -> Union[CatBoostClassifier, CatBoostRegressor]:
# #     """
# #     Loads a CatBoost model from a file.
# 
# #     Parameters
# #     ----------
# #     lpath : str
# #         Path to the CatBoost model file (.cbm extension)
# #     **kwargs : dict
# #         Additional keyword arguments passed to load_model method
# 
# #     Returns
# #     -------
# #     Union[CatBoostClassifier, CatBoostRegressor]
# #         Loaded CatBoost model object
# 
# #     Raises
# #     ------
# #     ValueError
# #         If file extension is not .cbm
# #     FileNotFoundError
# #         If model file does not exist
# 
# #     Examples
# #     --------
# #     >>> model = _load_catboost('model.cbm')
# #     >>> predictions = model.predict(X_test)
# #     """
# #     if not lpath.endswith(".cbm"):
# #         raise ValueError("File must have .cbm extension")
# 
# #     try:
# #         model = CatBoostClassifier().load_model(lpath, **kwargs)
# #     except:
# #         model = CatBoostRegressor().load_model(lpath, **kwargs)
# 
# #     return model
# 
# 
# # # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_load_modules/_catboost.py
# --------------------------------------------------------------------------------
