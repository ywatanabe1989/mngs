# Source code from: /home/ywatanabe/proj/mngs_dev/src/mngs/ai/plt/_optuna_study.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-03-30 08:24:55 (ywatanabe)"
# import os
# 
# 
# def optuna_study(lpath, value_str, sort=False):
#     """
#     Loads an Optuna study and generates various visualizations for each target metric.
# 
#     Parameters:
#     - lpath (str): Path to the Optuna study database.
#     - value_str (str): The name of the column to be used as the optimization target.
# 
#     Returns:
#     - None
#     """
#     import matplotlib
# 
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt
#     import mngs
#     import optuna
#     import pandas as pd
# 
#     plt, CC = mngs.plt.configure_mpl(plt, fig_scale=3)
# 
#     lpath = lpath.replace("./", "/")
# 
#     study = optuna.load_study(study_name=None, storage=lpath)
# 
#     sdir = lpath.replace("sqlite:///", "./").replace(".db", "/")
# 
#     # To get the best trial:
#     best_trial = study.best_trial
#     print(f"Best trial number: {best_trial.number}")
#     print(f"Best trial value: {best_trial.value}")
#     print(f"Best trial parameters: {best_trial.params}")
#     print(f"Best trial user attributes: {best_trial.user_attrs}")
# 
#     # Merge the user attributes into the study history DataFrame
#     study_history = study.trials_dataframe().rename(
#         columns={"value": value_str}
#     )
# 
#     if sort:
#         ascending = "MINIMIZE" in str(study.directions[0])  # [REVISED]
#         study_history = study_history.sort_values(
#             [value_str], ascending=ascending
#         )
# 
#     # Add user attributes to the study history DataFrame
#     attrs_df = []
#     for trial in study.trials:
#         user_attrs = trial.user_attrs
#         user_attrs = {k: v for k, v in user_attrs.items()}
#         attrs_df.append({"number": trial.number, **user_attrs})
#     attrs_df = pd.DataFrame(attrs_df).set_index("number")
# 
#     # Updates study history
#     study_history = study_history.merge(
#         attrs_df, left_index=True, right_index=True, how="left"
#     ).set_index("number")
#     try:
#         study_history = mngs.gen.mv_col(study_history, "SDIR", 1)
#         study_history["SDIR"] = study_history["SDIR"].apply(
#             lambda x: str(x).replace("RUNNING", "FINISHED")
#         )
#         best_trial_dir = study_history["SDIR"].iloc[0]
#         mngs.gen.symlink(best_trial_dir, sdir + "best_trial", force=True)
#     except Exception as e:
#         print(e)
#     mngs.io.save(study_history, sdir + "study_history.csv")
#     print(study_history)
# 
#     # To visualize the optimization history:
#     fig = optuna.visualization.plot_optimization_history(
#         study, target_name=value_str
#     )
#     mngs.io.save(fig, sdir + "optimization_history.png")
#     mngs.io.save(fig, sdir + "optimization_history.html")
#     plt.close()
# 
#     # To visualize the parameter importances:
#     fig = optuna.visualization.plot_param_importances(
#         study, target_name=value_str
#     )
#     mngs.io.save(fig, sdir + "param_importances.png")
#     mngs.io.save(fig, sdir + "param_importances.html")
#     plt.close()
# 
#     # To visualize the slice of the study:
#     fig = optuna.visualization.plot_slice(study, target_name=value_str)
#     mngs.io.save(fig, sdir + "slice.png")
#     mngs.io.save(fig, sdir + "slice.html")
#     plt.close()
# 
#     # To visualize the contour plot of the study:
#     fig = optuna.visualization.plot_contour(study, target_name=value_str)
#     mngs.io.save(fig, sdir + "contour.png")
#     mngs.io.save(fig, sdir + "contour.html")
#     plt.close()
# 
#     # To visualize the parallel coordinate plot of the study:
#     fig = optuna.visualization.plot_parallel_coordinate(
#         study, target_name=value_str
#     )
#     mngs.io.save(fig, sdir + "parallel_coordinate.png")
#     mngs.io.save(fig, sdir + "parallel_coordinate.html")
#     plt.close()
# 
# 
# if __name__ == "__main__":
#     mngs.plt.configure_mpl(plt, fig_scale=3)
#     lpath = "sqlite:///scripts/ml/clf/sub_conv_transformer_optuna/optuna_studies/optuna_study_v001.db"
#     lpath = "sqlite:///scripts/ml/clf/rocket_optuna/optuna_studies/optuna_study_v001.db"
#     optuna_study(lpath, "Validation bACC")
#     # scripts/ml/clf/sub_conv_transformer/optuna_studies/optuna_study_v032
# 
#     lpath = "sqlite:///scripts/ml/clf/sub_conv_transformer_optuna/optuna_studies/optuna_study_v020.db"
#     mngs.ml.plt.optuna_study(lpath, "val_loss", sort=True)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Add source code to the top of Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from mngs.ai.plt._optuna_study import *

class TestMainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        pass

    def teardown_method(self):
        # Clean up after tests
        pass

    def test_basic_functionality(self):
        # Basic test case
        raise NotImplementedError("Test not yet implemented")

    def test_edge_cases(self):
        # Edge case testing
        raise NotImplementedError("Test not yet implemented")

    def test_error_handling(self):
        # Error handling testing
        raise NotImplementedError("Test not yet implemented")

if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
