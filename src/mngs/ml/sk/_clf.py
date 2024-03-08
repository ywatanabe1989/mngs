#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-03-05 16:27:11 (ywatanabe)"

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sktime.classification.deep_learning.cnn import CNNClassifier
from sktime.classification.deep_learning.inceptiontime import (
    InceptionTimeClassifier,
)
from sktime.classification.deep_learning.lstmfcn import LSTMFCNClassifier
from sktime.classification.dummy import DummyClassifier
from sktime.classification.feature_based import TSFreshClassifier
from sktime.classification.hybrid import HIVECOTEV2
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.classification.kernel_based import RocketClassifier, TimeSeriesSVC
from sktime.transformations.panel.reduce import Tabularizer
from sktime.transformations.panel.rocket import Rocket


# rocket_pipeline = make_pipeline(
#     Rocket(n_jobs=-1),
#     RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
# )
def rocket_pipeline(*args, **kwargs):
    return make_pipeline(
        # Rocket(n_jobs=-1),
        Rocket(*args, **kwargs),
        SVC(probability=True, kernel="linear"),
    )


GB_pipeline = make_pipeline(
    Tabularizer(),
    GradientBoostingClassifier(),
)
