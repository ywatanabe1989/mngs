# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-03-23 17:36:05 (ywatanabe)"
# 
# import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
# from sklearn.feature_selection import SelectKBest, f_classif
# from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
# from sklearn.pipeline import make_pipeline
# from sklearn.svm import SVC, LinearSVC
# from sktime.classification.deep_learning.cnn import CNNClassifier
# from sktime.classification.deep_learning.inceptiontime import (
#     InceptionTimeClassifier,
# )
# from sktime.classification.deep_learning.lstmfcn import LSTMFCNClassifier
# from sktime.classification.dummy import DummyClassifier
# from sktime.classification.feature_based import TSFreshClassifier
# from sktime.classification.hybrid import HIVECOTEV2
# from sktime.classification.interval_based import TimeSeriesForestClassifier
# from sktime.classification.kernel_based import RocketClassifier, TimeSeriesSVC
# from sktime.transformations.panel.reduce import Tabularizer
# from sktime.transformations.panel.rocket import Rocket
# 
# # _rocket_pipeline = make_pipeline(
# #     Rocket(n_jobs=-1),
# #     RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
# # )
# 
# 
# # def rocket_pipeline(*args, **kwargs):
# #     return _rocket_pipeline
# 
# 
# def rocket_pipeline(*args, **kwargs):
#     return make_pipeline(
#         Rocket(*args, **kwargs),
#         LogisticRegression(
#             max_iter=1000
#         ),  # Increase max_iter if needed for convergence
#         # RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
#         # SVC(probability=True, kernel="linear"),
#     )
# 
# 
# # def rocket_pipeline(*args, **kwargs):
# #     return make_pipeline(
# #         Rocket(*args, **kwargs),
# #         SelectKBest(f_classif, k=500),
# #         PCA(n_components=100),
# #         LinearSVC(dual=False, tol=1e-3, C=0.1, probability=True),
# #     )
# 
# 
# GB_pipeline = make_pipeline(
#     Tabularizer(),
#     GradientBoostingClassifier(),
# )

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

from mngs.ai.sk._clf import *

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
