# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-07 18:56:57 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/ai/plt/aucs/example.py
# 
# import matplotlib.pyplot as plt
# import mngs
# import numpy as np
# from sklearn import datasets, svm
# from sklearn.model_selection import train_test_split
# from .roc_auc import roc_auc
# from .pre_rec_auc import pre_rec_auc
# 
# ################################################################################
# ## MNIST
# ################################################################################
# 
# digits = datasets.load_digits()
# 
# # flatten the images
# n_samples = len(digits.images)
# data = digits.images.reshape((n_samples, -1))
# 
# # Create a classifier: a support vector classifier
# clf = svm.SVC(gamma=0.001, probability=True)
# 
# # Split data into 50% train and 50% test subsets
# X_train, X_test, y_train, y_test = train_test_split(
#     data, digits.target, test_size=0.5, shuffle=False
# )
# 
# # Learn the digits on the train subset
# clf.fit(X_train, y_train)
# 
# # Predict the value of the digit on the test subset
# predicted_proba = clf.predict_proba(X_test)
# predicted = clf.predict(X_test)
# 
# n_classes = len(np.unique(digits.target))
# labels = ["Class {}".format(i) for i in range(n_classes)]
# 
# ## Configures matplotlib
# plt.rcParams["font.size"] = 20
# plt.rcParams["legend.fontsize"] = "xx-small"
# scale = 0.75
# plt.rcParams["figure.figsize"] = (16 * scale, 9 * scale)
# 
# ################################################################################
# ## Main
# ################################################################################
# ## ROC Curve
# fig_roc, metrics_roc = roc_auc(plt, y_test, predicted_proba, labels)
# fig_roc.show()
# ## Precision-Recall Curve
# fig_pre_rec, metrics_pre_rec = pre_rec_auc(
#     plt, y_test, predicted_proba, labels
# )
# fig_pre_rec.show()
# 
# #
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

from ...src.mngs..ai.plt.aucs.example import *

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
