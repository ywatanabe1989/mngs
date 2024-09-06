#!/usr/bin/env python3

import os
import random
import sys
from collections import defaultdict
from pprint import pprint

import matplotlib
import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
)
from glob import glob

class MultiClassificationReporter(object):
    """
    A class for reporting classification metrics for multiple targets.

    This class manages multiple ClassificationReporter instances, one for each target.

    Attributes:
        tgt2id (dict): Mapping of target names to their indices.
        reporters (list): List of ClassificationReporter instances, one for each target.
    """

    def __init__(self, sdir, tgts=None):
        """
        Initialize the MultiClassificationReporter.

        Args:
            sdir (str): Base directory for saving results.
            tgts (list): List of target names. If None, a single unnamed target is assumed.
        """
        if tgts is None:
            sdirs = [""]
        else:
            sdirs = [os.path.join(sdir, tgt, "/") for tgt in tgts]
        sdirs = [sdir + tgt + "/" for tgt in tgts]

        self.tgt2id = {tgt: i_tgt for i_tgt, tgt in enumerate(tgts)}
        self.reporters = [ClassificationReporter(sdir) for sdir in sdirs]

    def add(self, obj_name, obj, tgt=None):
        """
        Add an object to the reporter for a specific target.

        Args:
            obj_name (str): Name of the object to add.
            obj: The object to add (can be a figure, DataFrame, or scalar).
            tgt (str): The target name.
        """
        i_tgt = self.tgt2id[tgt]
        self.reporters[i_tgt].add(obj_name, obj)

    def calc_metrics(
        self,
        true_class,
        pred_class,
        pred_proba,
        labels=None,
        i_fold=None,
        show=True,
        auc_plt_config=dict(
            figsize=(7, 7),
            labelsize=8,
            fontsize=7,
            legendfontsize=6,
            tick_size=0.8,
            tick_width=0.2,
        ),
        tgt=None,
    ):
        """
        Calculate classification metrics for a specific target.

        Args:
            true_class (array-like): True class labels.
            pred_class (array-like): Predicted class labels.
            pred_proba (array-like): Predicted probabilities.
            labels (list): List of class labels.
            i_fold (int): Fold number (for cross-validation).
            show (bool): Whether to display results.
            auc_plt_config (dict): Configuration for AUC plot.
            tgt (str): The target name.
        """
        i_tgt = self.tgt2id[tgt]
        self.reporters[i_tgt].calc_metrics(
            true_class,
            pred_class,
            pred_proba,
            labels=labels,
            i_fold=i_fold,
            show=show,
            auc_plt_config=auc_plt_config,
        )

    def summarize(
        self,
        n_round=3,
        show=False,
        tgt=None,
    ):
        """
        Summarize the classification results for a specific target.

        Args:
            n_round (int): Number of decimal places to round to.
            show (bool): Whether to display the summary.
            tgt (str): The target name.
        """
        i_tgt = self.tgt2id[tgt]
        self.reporters[i_tgt].summarize(
            n_round=n_round,
            show=show,
        )

    def save(
        self,
        files_to_reproduce=None,
        meta_dict=None,
        tgt=None,
    ):
        """
        Save the classification results for a specific target.

        Args:
            files_to_reproduce (list): List of files needed to reproduce the results.
            meta_dict (dict): Additional metadata to save.
            tgt (str): The target name.
        """
        i_tgt = self.tgt2id[tgt]
        self.reporters[i_tgt].save(
            files_to_reproduce=files_to_reproduce,
            meta_dict=meta_dict,
        )

    def plot_and_save_conf_mats(
        self,
        plt,
        extend_ratio=1.0,
        colorbar=True,
        confmat_plt_config=None,
        sci_notation_kwargs=None,
        tgt=None,
    ):
        """
        Plot and save confusion matrices for a specific target.

        Args:
            plt: Matplotlib pyplot object.
            extend_ratio (float): Ratio to extend the plot.
            colorbar (bool): Whether to include a colorbar.
            confmat_plt_config (dict): Configuration for confusion matrix plot.
            sci_notation_kwargs (dict): Keywords for scientific notation.
            tgt (str): The target name.
        """
        i_tgt = self.tgt2id[tgt]
        self.reporters[i_tgt].plot_and_save_conf_mats(
            plt,
            extend_ratio=extend_ratio,
            colorbar=colorbar,
            confmat_plt_config=confmat_plt_config,
            sci_notation_kwargs=sci_notation_kwargs,
        )


class ClassificationReporter(object):
    """
    A class for reporting various classification metrics and saving them.

    This class calculates and saves the following metrics:
    - Balanced Accuracy
    - Matthews Correlation Coefficient (MCC)
    - Confusion Matrix
    - Classification Report
    - ROC AUC score / curve
    - Precision-Recall AUC score / curve

    Attributes:
        sdir (str): Directory to save the results.
        folds_dict (defaultdict): Dictionary to store results for each fold.
    """

    def __init__(self, sdir):
        """
        Initialize the ClassificationReporter.

        Args:
            sdir (str): Directory to save the results.
        """
        self.sdir = sdir
        self.folds_dict = defaultdict(list)
        mngs.general.fix_seeds(os=os, random=random, np=np, torch=torch, show=False)

    def add(
        self,
        obj_name,
        obj,
    ):
        """
        Add an object to the reporter.

        This method can be used to add figures, DataFrames, or scalars to the report.

        Args:
            obj_name (str): Name of the object to add.
            obj: The object to add (can be a figure, DataFrame, or scalar).

        Example:
            fig, ax = plt.subplots()
            ax.plot(np.random.rand(10))
            reporter.add("manu_figs", fig)

            df = pd.DataFrame(np.random.rand(5, 3))
            reporter.add("manu_dfs", df)

            scalar = random.random()
            reporter.add("manu_scalers", scalar)
        """
        assert isinstance(obj_name, str)
        self.folds_dict[obj_name].append(obj)

    @staticmethod
    def calc_bACC(true_class, pred_class, i_fold, show=False):
        """
        Calculate Balanced Accuracy.

        Args:
            true_class (array-like): True class labels.
            pred_class (array-like): Predicted class labels.
            i_fold (int): Fold number (for cross-validation).
            show (bool): Whether to display the result.

        Returns:
            float: Balanced accuracy score.
        """
        balanced_acc = balanced_accuracy_score(true_class, pred_class)
        if show:
            print(f"
Balanced ACC in fold#{i_fold} was {balanced_acc:.3f}
")
        return balanced_acc

    @staticmethod
    def calc_mcc(true_class, pred_class, i_fold, show=False):
        """
        Calculate Matthews Correlation Coefficient (MCC).

        Args:
            true_class (array-like): True class labels.
            pred_class (array-like): Predicted class labels.
            i_fold (int): Fold number (for cross-validation).
            show (bool): Whether to display the result.

        Returns:
            float: Matthews Correlation Coefficient.
        """
        mcc = float(matthews_corrcoef(true_class, pred_class))
        if show:
            print(f"
MCC in fold#{i_fold} was {mcc:.3f}
")
        return mcc

    @staticmethod
    def calc_conf_mat(true_class, pred_class, labels, i_fold, show=False):
        """
        Calculate Confusion Matrix.

        This method assumes unique classes of true_class and pred_class are the same.

        Args:
            true_class (array-like): True class labels.
            pred_class (array-like): Predicted class labels.
            labels (list): List of class labels.
            i_fold (int): Fold number (for cross-validation).
            show (bool): Whether to display the result.

        Returns:
            pd.DataFrame: Confusion matrix as a DataFrame.
        """
        conf_mat = pd.DataFrame(
            data=confusion_matrix(
                true_class, pred_class, labels=np.arange(len(labels))
            ),
            columns=labels,
        ).set_index(pd.Series(list(labels)))

        if show:
            print(f"
Confusion Matrix in fold#{i_fold}")
            print(conf_mat)
            print()

        return conf_mat

    # ... [rest of the methods with similar docstring improvements] ...

if __name__ == "__main__":
    # ... [example usage code] ...
    pass
