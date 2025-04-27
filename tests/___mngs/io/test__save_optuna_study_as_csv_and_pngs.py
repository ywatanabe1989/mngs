# Source code from: /home/ywatanabe/proj/mngs_dev/src/mngs/io/_save_optuna_study_as_csv_and_pngs.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 17:01:15 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/io/_save_optuna_study_as_csv_and_pngs.py
# 
# def save_optuna_study_as_csv_and_pngs(study, sdir):
#     import optuna
# 
#     ## Trials DataFrame
#     trials_df = study.trials_dataframe()
# 
#     ## Figures
#     hparams_keys = list(study.best_params.keys())
#     slice_plot = optuna.visualization.plot_slice(study, params=hparams_keys)
#     contour_plot = optuna.visualization.plot_contour(
#         study, params=hparams_keys
#     )
#     optim_hist_plot = optuna.visualization.plot_optimization_history(study)
#     parallel_coord_plot = optuna.visualization.plot_parallel_coordinate(
#         study, params=hparams_keys
#     )
#     hparam_importances_plot = optuna.visualization.plot_param_importances(
#         study
#     )
#     figs_dict = dict(
#         slice_plot=slice_plot,
#         contour_plot=contour_plot,
#         optim_hist_plot=optim_hist_plot,
#         parallel_coord_plot=parallel_coord_plot,
#         hparam_importances_plot=hparam_importances_plot,
#     )
# 
#     ## Saves
#     save(trials_df, sdir + "trials_df.csv")
# 
#     for figname, fig in figs_dict.items():
#         save(fig, sdir + f"{figname}.png")
# 
# 
# # EOF

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

from mngs.io._save_optuna_study_as_csv_and_pngs import *

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
