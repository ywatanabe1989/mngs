#!/usr/bin/env python3

import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import mngs
import yaml
from natsort import natsorted
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)


class Reporter:
    """Saves the following metrics under sdir.
       - Balanced Accuracy
       - MCC
       - Confusion Matrix
       - Classification Report
       - ROC AUC score / curve
       - PRE-REC AUC score / curve

    Manual adding example:
        ##############################
        ## fig
        ##############################
        fig, ax = plt.subplots()
        ax.plot(np.random.rand(10))
        reporter.add(
            "manu_figs",
            fig,
            {
                "dirname": "manu_fig_dir/",
                "ext": ".png",
            },
        )
        ##############################
        ## DataFrame
        ##############################
        df = pd.DataFrame(np.random.rand(5, 3))
        reporter.add("manu_dfs", df, {"fname": "manu_dfs.csv", "method": "mean"})
        ##############################
        ## scalar
        ##############################
        scalar = random.random()
        reporter.add(
            "manu_scalars",
            scalar,
            {"fname": "manu_scalars.csv", "column_name": "manu_column_name"},
        )
    """

    def __init__(self, sdir):
        self.sdir = sdir
        self.ts = mngs.general.TimeStamper()

        self.balanced_accs_folds = []
        self.mccs_folds = []
        self.conf_mats_folds = []
        self.clf_reports_folds = []
        self.roc_aucs_macro_folds = []
        self.roc_aucs_micro_folds = []
        self.roc_aucs_figs_folds = []
        self.pre_rec_aucs_macro_folds = []
        self.pre_rec_aucs_micro_folds = []
        self.pre_rec_aucs_figs_folds = []

        ## for more flexible collection
        self.added_folds_obj_dict = mngs.general.listed_dict()
        self.added_folds_meta_dict = {}
        self._added_at_least_once = False

        print("\n{}\n".format(mngs.general.gen_timestamp()))
        self.ts("\nReporter has been initialized.\n")

    def add(
        self,
        obj_name,
        obj,
        meta_dict,
    ):

        ## Initializes the list
        if not self._added_at_least_once:
            self.added_folds_obj_dict[obj_name] = []
        self._added_at_least_once = True

        ## adds
        self.added_folds_obj_dict[obj_name].append(obj)
        self.added_folds_meta_dict[obj_name] = meta_dict

    def calc_metrics(
        self,
        true_class,
        pred_class,
        pred_proba,
        labels=None,
        i_fold=None,
        show=True,
        plot=False,
    ):
        """Calculates ACC, Confusion Matrix, Classification Report, and ROC-AUC score."""
        self.labels = labels

        true_class, pred_class, pred_proba = (
            mngs.general.torch_to_arr(true_class),
            mngs.general.torch_to_arr(pred_class),
            mngs.general.torch_to_arr(pred_proba),
        )

        ####################
        ## ACC
        ####################
        acc = (true_class.reshape(-1) == pred_class.reshape(-1)).mean()
        if show:
            print("\nACC in fold#{} was {:.3f}\n".format(i_fold, acc))
        balanced_acc = balanced_accuracy_score(
            true_class.reshape(-1), pred_class.reshape(-1)
        )
        if show:
            print("\nBalanced ACC in fold#{} was {:.3f}\n".format(i_fold, balanced_acc))

        ####################
        ## MCC
        ####################
        mcc = matthews_corrcoef(true_class.reshape(-1), pred_class.reshape(-1))
        if show:
            print("\nMCC in fold#{} was {:.3f}\n".format(i_fold, mcc))

        ####################
        ## Confusion Matrix
        ####################
        conf_mat = confusion_matrix(true_class, pred_class)
        conf_mat = pd.DataFrame(data=conf_mat, columns=labels).set_index(
            pd.Series(list(labels))
        )
        if show:
            print("\nConfusion Matrix in fold#{}: \n{}\n".format(i_fold, conf_mat))

        ####################
        ## Classification Report
        ####################
        clf_report = pd.DataFrame(
            classification_report(
                true_class,
                pred_class,
                target_names=labels,
                output_dict=True,
            )
        )
        # accuracy to balanced accuracy
        clf_report["accuracy"] = balanced_acc
        clf_report = clf_report.rename(columns={"accuracy": "balanced accuracy"})
        clf_report = clf_report.round(3)
        # rename 'support' to 'sample size'
        clf_report["index"] = clf_report.index
        clf_report.loc["support", "index"] = "sample size"
        clf_report.set_index("index", drop=True, inplace=True)
        clf_report.index.name = None
        if show:
            print(
                "\nClassification Report in fold#{}: \n{}\n".format(i_fold, clf_report)
            )

        ####################
        ## ROC-AUC score
        ####################
        mngs.plt.configure_mpl(
            plt,
            figsize=(7, 7),
            labelsize=8,
            fontsize=7,
            legendfontsize=6,
            tick_size=0.8,
            tick_width=0.2,
        )

        fig_roc, metrics_roc_auc_dict = mngs.ml.plt.roc_auc(
            plt,
            true_class,
            pred_proba,
            labels,
        )
        if plot:
            fig_roc.plot()
        else:
            plt.close()

        ####################
        ## PRE-REC AUC score
        ####################
        fig_pre_rec, metrics_pre_rec_auc_dict = mngs.ml.plt.pre_rec_auc(
            plt, true_class, pred_proba, labels
        )
        if plot:
            fig_pre_rec.plot()
        else:
            plt.close()

        ####################
        # to buffer
        ####################
        self.mccs_folds.append(mcc)
        self.balanced_accs_folds.append(balanced_acc)
        self.conf_mats_folds.append(conf_mat)
        self.clf_reports_folds.append(clf_report)
        self.roc_aucs_micro_folds.append(metrics_roc_auc_dict["roc_auc"]["micro"])
        self.roc_aucs_macro_folds.append(metrics_roc_auc_dict["roc_auc"]["macro"])
        self.roc_aucs_figs_folds.append(fig_roc)
        self.pre_rec_aucs_micro_folds.append(
            metrics_pre_rec_auc_dict["pre_rec_auc"]["micro"]
        )
        self.pre_rec_aucs_macro_folds.append(
            metrics_pre_rec_auc_dict["pre_rec_auc"]["macro"]
        )
        self.pre_rec_aucs_figs_folds.append(fig_pre_rec)
        del fig_roc, fig_pre_rec

    def summarize(
        self,
    ):
        ####################
        ## Summarize each fold's metirics
        ####################
        self.mcc_cv_mean, self.mcc_cv_std = self.take_mean_and_std(self.mccs_folds)
        self.balanced_acc_cv_mean, self.balanced_acc_cv_std = self.take_mean_and_std(
            self.balanced_accs_folds
        )
        self.conf_mat_cv_sum = self.summarize_dfs(self.conf_mats_folds, method="sum")
        self.clf_report_cv_mean, self.clf_report_cv_std = self.summarize_dfs(
            self.clf_reports_folds, method="mean"
        )
        self.roc_auc_micro_cv_mean, self.roc_auc_micro_cv_std = self.take_mean_and_std(
            self.roc_aucs_micro_folds
        )
        self.roc_auc_macro_cv_mean, self.roc_auc_macro_cv_std = self.take_mean_and_std(
            self.roc_aucs_macro_folds
        )
        (
            self.pre_rec_auc_micro_cv_mean,
            self.pre_rec_auc_micro_cv_std,
        ) = self.take_mean_and_std(self.pre_rec_aucs_micro_folds)
        (
            self.pre_rec_auc_macro_cv_mean,
            self.pre_rec_auc_macro_cv_std,
        ) = self.take_mean_and_std(self.pre_rec_aucs_macro_folds)

        self.num_folds = len(self.conf_mats_folds)
        self.print_metrics()

    def print_metrics(self):
        print("\n --- {}-fold CV overall metrics --- \n".format(self.num_folds))
        print(
            "\nThe Mattews correlation coefficient: {} +/- {} (mean +/- std.; n={})\n".format(
                self.mcc_cv_mean, self.mcc_cv_std, self.num_folds
            )
        )

        print(
            "\nBalanced Accuracy Score: {} +/- {} (mean +/- std.; n={})\n".format(
                self.balanced_acc_cv_mean, self.balanced_acc_cv_std, self.num_folds
            )
        )

        print(
            "\nConfusion Matrix (Test; sum; num. folds={})\n{}\n".format(
                self.num_folds, self.conf_mat_cv_sum
            )
        )
        print(
            "\nClassification Report (Test; mean; num. folds={})\n{}\n".format(
                self.num_folds, self.clf_report_cv_mean
            )
        )
        print(
            "\nClassification Report (Test; std; num. folds={})\n{}\n".format(
                self.num_folds, self.clf_report_cv_std
            )
        )
        print(
            "\nROC AUC micro Score: {} +/- {} (mean +/- std.; n={})\n".format(
                self.roc_auc_micro_cv_mean, self.roc_auc_micro_cv_std, self.num_folds
            )
        )
        print(
            "\nROC AUC macro Score: {} +/- {} (mean +/- std.; n={})\n".format(
                self.roc_auc_macro_cv_mean, self.roc_auc_macro_cv_std, self.num_folds
            )
        )
        print(
            "\nPrecision-Recall AUC micro Score: {} +/- {} (mean +/- std.; n={})\n".format(
                self.pre_rec_auc_micro_cv_mean,
                self.pre_rec_auc_micro_cv_std,
                self.num_folds,
            )
        )
        print(
            "\nPrecision-Recall AUC macro Score: {} +/- {} (mean +/- std.; n={})\n".format(
                self.pre_rec_auc_macro_cv_mean,
                self.pre_rec_auc_macro_cv_std,
                self.num_folds,
            )
        )

    def take_mean_and_std(self, obj_list, n_round=3):
        arr = np.array(obj_list)
        return arr.mean(axis=0).round(n_round), arr.std(axis=0).round(n_round)

    def save_listed_scalars(
        self,
        listed_scalars,
        fname=None,
        column_name=None,
        n_round=3,
        show=True,
        makedirs=False,
    ):
        cat = [
            np.mean(listed_scalars).round(n_round),
            np.std(listed_scalars).round(n_round),
        ] + listed_scalars

        num_folds = len(listed_scalars)

        indi_suffix_cat = (
            ["{}-fold CV mean".format(num_folds)]
            + ["{}-fold CV std.".format(num_folds)]
            + ["fold#{}".format(i) for i in range(num_folds)]
        )

        mngs.general.save(
            cat,
            self.sdir + fname,
            indi_suffix=indi_suffix_cat,
            column_name=column_name,
            show=show,
            makedirs=makedirs,
        )

    def save_listed_dfs(
        self,
        listed_dfs,
        method="mean",
        fname=None,
        n_round=3,
        show=True,
        makedirs=False,
    ):
        if method == "mean":
            summarized_dfs = self.summarize_dfs(
                listed_dfs, method=method, n_round=n_round
            )
            indi_suffix_cat = (
                ["{}-fold CV mean".format(self.num_folds)]
                + ["{}-fold CV std.".format(self.num_folds)]
                + ["fold#{}".format(i) for i in range(self.num_folds)]
            )

        if method == "sum":
            summarized_dfs = self.summarize_dfs(
                listed_dfs, method=method, n_round=n_round
            )
            summarized_dfs = [summarized_dfs]
            indi_suffix_cat = ["sum-fold CV mean".format(self.num_folds)] + [
                "fold#{}".format(i) for i in range(self.num_folds)
            ]

        cat = [*summarized_dfs] + listed_dfs

        mngs.general.save(
            cat,
            self.sdir + fname,
            indi_suffix=indi_suffix_cat,
            overwrite=True,
            show=show,
            makedirs=makedirs,
        )

    def summarize_dfs(self, dfs_list, method="sum", n_round=3):
        df_zero = 0 * dfs_list[0].copy()  # get the table format

        if method == "sum":
            return df_zero + np.array(dfs_list).sum(axis=0).round(n_round)

        if method == "mean":
            arr = np.array(dfs_list)
            df_mean = arr.astype(float).mean(axis=0).round(n_round)
            df_std = arr.astype(float).std(axis=0).round(n_round)
            return df_zero + df_mean, df_zero + df_std

    def save_listed_figures(
        self, listed_figures, dirname="None", ext="png", show=True, makedirs=False
    ):
        for i_fold, obj in enumerate(listed_figures):  # range(len(listed_figures)):
            spath = self.sdir + dirname + "fold#{}".format(i_fold) + ext
            mngs.general.save(obj, spath, show=show, makedirs=makedirs)
        # print("\nSaved to: {s}\n".format(s=dirname))

    def save(self, others_dict=None, makedirs=True):
        # if makedirs:
        #     os.makedirs(self.sdir, exist_ok=True)

        ####################
        ## Others dict
        ####################
        if isinstance(others_dict, dict):
            for sfname, obj in others_dict.items():
                mngs.general.save(obj, self.sdir + sfname, makedirs=makedirs)

        ####################
        ## MCC
        ####################
        self.save_listed_scalars(
            self.mccs_folds,
            fname="mccs.csv",
            column_name="The Mattews corr. coeff.",
            n_round=3,
            makedirs=makedirs,
        )

        ####################
        ## Balanced ACC
        ####################
        self.save_listed_scalars(
            self.balanced_accs_folds,
            fname="balanced_accs.csv",
            column_name="Balanced Accuracy",
            n_round=3,
            makedirs=makedirs,
        )

        ####################
        ## Confusion Matrix
        ####################
        mngs.plt.configure_mpl(
            plt,
            figsize=(8, 8),
            labelsize=8,
            fontsize=6,
            legendfontsize=6,
            tick_size=0.8,
            tick_width=0.2,
        )

        # each fold
        self.conf_mats_folds_fig = [
            mngs.ml.plt.confusion_matrix(
                plt,
                cm,
                labels=self.labels,
                title="Test Mouse#{}".format(i_cm + 1),
                extend_ratio=0.4,
                colorbar=True,
            )
            for i_cm, cm in enumerate(self.conf_mats_folds)
        ]

        for fig_cm_fold in self.conf_mats_folds_fig:
            fig_cm_fold.axes[-1] = mngs.plt.ax_scientific_notation(
                fig_cm_fold.axes[-1],
                3,
                fformat="%3.1f",
                y=True,
            )

        for i_fold, cm in enumerate(self.conf_mats_folds_fig):
            mngs.general.save(
                cm, self.sdir + "conf_mat/" + "fold#{}.png".format(i_fold)
            )
        self.save_listed_dfs(
            self.conf_mats_folds,
            method="sum",
            fname="conf_mat/conf_mats.csv",
            makedirs=makedirs,
        )
        # kCV overall-sum
        title = "{}-CV overall-sum".format(len(self.conf_mats_folds))

        fig_cm = mngs.ml.plt.confusion_matrix(
            plt,
            self.conf_mat_cv_sum,
            labels=self.labels,
            title=title,
            extend_ratio=0.4,
            colorbar=True,
        )

        # fixme
        fig_cm.axes[-1] = mngs.plt.ax_scientific_notation(
            fig_cm.axes[-1],
            3,
            fformat="%3.1f",
            y=True,
        )

        mngs.general.save(
            fig_cm, self.sdir + "conf_mat/overall_sum.png", makedirs=makedirs
        )

        ####################
        ## Classification Report
        ####################
        self.save_listed_dfs(
            self.clf_reports_folds,
            method="mean",
            fname="clf_reports.csv",
            makedirs=makedirs,
        )

        ####################
        ## ROC-AUC and PRE-REC AUC
        ####################
        _roc_aucs_micro_cat = [
            self.roc_auc_micro_cv_mean,
            self.roc_auc_micro_cv_std,
        ] + self.roc_aucs_micro_folds
        _roc_aucs_macro_cat = [
            self.roc_auc_macro_cv_mean,
            self.roc_auc_macro_cv_std,
        ] + self.roc_aucs_macro_folds
        _pre_rec_aucs_micro_cat = [
            self.pre_rec_auc_micro_cv_mean,
            self.pre_rec_auc_micro_cv_std,
        ] + self.pre_rec_aucs_micro_folds
        _pre_rec_aucs_macro_cat = [
            self.pre_rec_auc_macro_cv_mean,
            self.pre_rec_auc_macro_cv_std,
        ] + self.pre_rec_aucs_macro_folds
        _indi_suffix_cat = (
            ["{}-fold CV mean".format(self.num_folds)]
            + ["{}-fold CV std.".format(self.num_folds)]
            + ["fold#{}".format(i) for i in range(self.num_folds)]
        )

        aucs_df = pd.DataFrame(
            {
                "ROC AUC micro": _roc_aucs_micro_cat,
                "ROC AUC macro": _roc_aucs_macro_cat,
                "PRE-REC AUC micro": _pre_rec_aucs_micro_cat,
                "PRE-REC AUC macro": _pre_rec_aucs_macro_cat,
            }
        )
        aucs_df.index = _indi_suffix_cat
        mngs.general.save(aucs_df, self.sdir + "aucs.csv", makedirs=makedirs)

        for i_fold in range(len(self.roc_aucs_figs_folds)):
            spath = self.sdir + "roc_curves/fold#{}.png".format(i_fold)
            mngs.general.save(
                self.roc_aucs_figs_folds[i_fold], spath, makedirs=makedirs
            )

        for i_fold in range(len(self.pre_rec_aucs_figs_folds)):
            spath = self.sdir + "pre_rec_curves/fold#{}.png".format(i_fold)
            mngs.general.save(
                self.pre_rec_aucs_figs_folds[i_fold], spath, makedirs=makedirs
            )

        ## Manually added scalars, dfs, and figures
        for k in self.added_folds_obj_dict.keys():
            listed_obj = self.added_folds_obj_dict[k]
            meta = self.added_folds_meta_dict[k]

            if mngs.general.is_listed_X(listed_obj, [int, float]):
                self.save_listed_scalars(
                    listed_obj, show=True, makedirs=makedirs, **meta
                )

            if mngs.general.is_listed_X(listed_obj, pd.DataFrame):
                self.save_listed_dfs(listed_obj, show=True, makedirs=makedirs, **meta)

            # if is_listed_X(
            if mngs.general.is_listed_X(
                listed_obj,
                [plt.Figure, matplotlib.figure.Figure],
            ):
                self.save_listed_figures(
                    listed_obj, show=True, makedirs=makedirs, **meta
                )


if __name__ == "__main__":
    import random

    import numpy as np
    import mngs
    from catboost import CatBoostClassifier, Pool
    from sklearn.datasets import load_digits
    from sklearn.model_selection import StratifiedKFold

    mngs.general.fix_seeds(np=np)

    ## Loads
    mnist = load_digits()
    X, T = mnist.data, mnist.target
    labels = mnist.target_names.astype(str)

    ## Make save dir
    sdir = "/tmp/sdir/"

    skf = StratifiedKFold(n_splits=5, shuffle=True)

    ## Main
    reporter = Reporter(sdir)
    for i_fold, (indi_tra, indi_tes) in enumerate(skf.split(X, T)):
        print(i_fold)
        ## koko
        X_tra, T_tra = X[indi_tra], T[indi_tra]
        X_tes, T_tes = X[indi_tes], T[indi_tes]

        clf = CatBoostClassifier(verbose=False)

        clf.fit(X_tra, T_tra, verbose=False)

        ## Prediction
        pred_proba_tes = clf.predict_proba(X_tes)
        pred_cls_tes = np.argmax(pred_proba_tes, axis=1)

        ##############################
        # fig
        ##############################
        fig, ax = plt.subplots()
        ax.plot(np.arange(10))
        reporter.add(
            "manu_figs",
            fig,
            {
                "dirname": "manu_fig_dir/",
                "ext": ".png",
            },
        )

        df = pd.DataFrame(np.random.rand(5, 3))
        reporter.add("manu_dfs", df, {"fname": "manu_dfs.csv", "method": "mean"})

        # scalar
        scalar = random.random()
        reporter.add(
            "manu_scalars",
            scalar,
            {"fname": "manu_scalars.csv", "column_name": "manu_column_name"},
        )

        ## Metrics
        reporter.calc_metrics(
            T_tes,
            pred_cls_tes,
            pred_proba_tes,
            labels=labels,
            i_fold=i_fold,
        )

    reporter.summarize()
    reporter.save()

    ## EOF
