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
    def __init__(self, sdir, tgts=None):
        if tgts is None:
            sdirs = [""]
        else:
            sdirs = [os.path.join(sdir, tgt, "/") for tgt in tgts]
        sdirs = [sdir + tgt + "/" for tgt in tgts]

        self.tgt2id = {tgt: i_tgt for i_tgt, tgt in enumerate(tgts)}
        self.reporters = [ClassificationReporter(sdir) for sdir in sdirs]

    def add(self, obj_name, obj, tgt=None):
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
        i_tgt = self.tgt2id[tgt]
        self.reporters[i_tgt].plot_and_save_conf_mats(
            plt,
            extend_ratio=extend_ratio,
            colorbar=colorbar,
            confmat_plt_config=confmat_plt_config,
            sci_notation_kwargs=sci_notation_kwargs,
        )


class ClassificationReporter(object):
    """Saves the following metrics under sdir.
       - Balanced Accuracy
       - MCC
       - Confusion Matrix
       - Classification Report
       - ROC AUC score / curve
       - PRE-REC AUC score / curve

    Example is described in this file.
    """

    def __init__(self, sdir):
        self.sdir = sdir
        self.folds_dict = defaultdict(list)
        mngs.general.fix_seeds(os=os, random=random, np=np, torch=torch, show=False)

    def add(
        self,
        obj_name,
        obj,
    ):
        """
        ## fig
        fig, ax = plt.subplots()
        ax.plot(np.random.rand(10))
        reporter.add("manu_figs", fig)

        ## DataFrame
        df = pd.DataFrame(np.random.rand(5, 3))
        reporter.add("manu_dfs", df)

        ## scalar
        scalar = random.random()
        reporter.add("manu_scalers", scalar)
        """
        assert isinstance(obj_name, str)
        self.folds_dict[obj_name].append(obj)

    @staticmethod
    def calc_bACC(true_class, pred_class, i_fold, show=False):
        """Balanced ACC"""
        balanced_acc = balanced_accuracy_score(true_class, pred_class)
        if show:
            print(f"\nBalanced ACC in fold#{i_fold} was {balanced_acc:.3f}\n")
        return balanced_acc

    @staticmethod
    def calc_mcc(true_class, pred_class, i_fold, show=False):
        """MCC"""
        mcc = float(matthews_corrcoef(true_class, pred_class))
        if show:
            print(f"\nMCC in fold#{i_fold} was {mcc:.3f}\n")
        return mcc

    @staticmethod
    def calc_conf_mat(true_class, pred_class, labels, i_fold, show=False):
        """
        Confusion Matrix
        This method assumes unique classes of true_class and pred_class are the same.
        """
        # conf_mat = pd.DataFrame(
        #     data=confusion_matrix(true_class, pred_class),
        #     columns=pred_labels,
        #     index=true_labels,
        # )

        conf_mat = pd.DataFrame(
            data=confusion_matrix(
                true_class, pred_class, labels=np.arange(len(labels))
            ),
            columns=labels,
        ).set_index(pd.Series(list(labels)))

        if show:
            print(f"\nConfusion Matrix in fold#{i_fold}: \n")
            pprint(conf_mat)
            print()

        return conf_mat

    @staticmethod
    def calc_clf_report(
        true_class, pred_class, labels, balanced_acc, i_fold, show=False
    ):
        """Classification Report"""
        clf_report = pd.DataFrame(
            classification_report(
                true_class,
                pred_class,
                labels=np.arange(len(labels)),
                target_names=labels,
                output_dict=True,
            )
        )

        # ACC to bACC
        clf_report["accuracy"] = balanced_acc
        clf_report = pd.concat(
            [clf_report[labels], clf_report[["accuracy", "macro avg", "weighted avg"]]],
            axis=1,
        )
        clf_report = clf_report.rename(columns={"accuracy": "balanced accuracy"})
        clf_report = clf_report.round(3)
        # Renames 'support' to 'sample size'
        clf_report["index"] = clf_report.index
        clf_report.loc["support", "index"] = "sample size"
        clf_report.set_index("index", drop=True, inplace=True)
        clf_report.index.name = None
        if show:
            print(f"\nClassification Report for fold#{i_fold}:\n")
            pprint(clf_report)
            print()
        return clf_report

    @staticmethod
    def calc_and_plot_roc_curve(true_class, pred_proba, labels, sdir_for_csv=None):
        # ROC-AUC
        fig_roc, metrics_roc_auc_dict = mngs.ml.plt.roc_auc(
            plt,
            true_class,
            pred_proba,
            labels,
            sdir_for_csv=sdir_for_csv,
        )
        plt.close()
        return fig_roc, metrics_roc_auc_dict

    @staticmethod
    def calc_and_plot_pre_rec_curve(true_class, pred_proba, labels):
        # PRE-REC AUC
        fig_pre_rec, metrics_pre_rec_auc_dict = mngs.ml.plt.pre_rec_auc(
            plt, true_class, pred_proba, labels
        )
        plt.close()
        return fig_pre_rec, metrics_pre_rec_auc_dict

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
    ):
        """
        Calculates ACC, Confusion Matrix, Classification Report, and ROC-AUC score on a fold.
        Metrics and curves will be kept in self.folds_dict.
        """

        ## Preparation
        # for convenience
        true_class = mngs.general.torch_to_arr(true_class).astype(int).reshape(-1)
        pred_class = (
            mngs.general.torch_to_arr(pred_class).astype(np.float64).reshape(-1)
        )
        pred_proba = mngs.general.torch_to_arr(pred_proba).astype(np.float64)

        # for curves
        mngs.plt.configure_mpl(
            plt,
            **auc_plt_config,
        )

        ## Calc metrics
        # Balanced ACC
        bacc = self.calc_bACC(true_class, pred_class, i_fold, show=show)
        self.folds_dict["balanced_acc"].append(bacc)

        # MCC
        self.folds_dict["mcc"].append(
            self.calc_mcc(true_class, pred_class, i_fold, show=show)
        )

        # Confusion Matrix
        self.folds_dict["conf_mat/conf_mat"].append(
            self.calc_conf_mat(
                true_class,
                pred_class,
                labels,
                i_fold,
                show=show,
            )
        )

        # Classification Report
        self.folds_dict["clf_report"].append(
            self.calc_clf_report(
                true_class, pred_class, labels, bacc, i_fold, show=show
            )
        )

        ## Curves
        # ROC curve
        self.sdir_for_roc_csv = f"{self.sdir}roc/csv/"
        fig_roc, metrics_roc_auc_dict = self.calc_and_plot_roc_curve(
            true_class,
            pred_proba,
            labels,
            sdir_for_csv=self.sdir_for_roc_csv + f"fold#{i_fold}/",
        )
        self.folds_dict["roc/micro"].append(metrics_roc_auc_dict["roc_auc"]["micro"])
        self.folds_dict["roc/macro"].append(metrics_roc_auc_dict["roc_auc"]["macro"])
        self.folds_dict["roc/figs"].append(fig_roc)

        # PRE-REC curve
        fig_pre_rec, metrics_pre_rec_auc_dict = self.calc_and_plot_pre_rec_curve(
            true_class, pred_proba, labels
        )
        self.folds_dict["pre_rec/micro"].append(
            metrics_pre_rec_auc_dict["pre_rec_auc"]["micro"]
        )
        self.folds_dict["pre_rec/macro"].append(
            metrics_pre_rec_auc_dict["pre_rec_auc"]["macro"]
        )
        self.folds_dict["pre_rec/figs"].append(fig_pre_rec)

    @staticmethod
    def _mk_cv_index(n_folds):
        return [
            f"{n_folds}-folds_CV_mean",
            f"{n_folds}-fold_CV_std",
        ] + [f"fold#{i_fold}" for i_fold in range(n_folds)]

    def summarize_roc(
        self,
    ):

        folds_dirs = glob(self.sdir_for_roc_csv + "fold#*")
        n_folds = len(folds_dirs)

        # get class names
        _csv_files = glob(os.path.join(folds_dirs[0], "*"))
        classes_str = [
            csv_file.split("/")[-1].split(".csv")[0] for csv_file in _csv_files
        ]

        # dfs_classes = []
        # take mean and std by each class
        for cls_str in classes_str:

            fpaths_cls = [
                os.path.join(fold_dir, f"{cls_str}.csv")
                for fold_dir in folds_dirs
            ]

            ys = []
            roc_aucs = []
            for fpath_cls in fpaths_cls:
                loaded_df = mngs.io.load(fpath_cls)
                ys.append(loaded_df["y"])
                roc_aucs.append(loaded_df["roc_auc"])
            ys = pd.concat(ys, axis=1)
            roc_aucs = pd.concat(roc_aucs, axis=1)

            df_cls = loaded_df[["x"]].copy()
            df_cls["y_mean"] = ys.mean(axis=1)
            df_cls["y_std"] = ys.std(axis=1)
            df_cls["roc_auc_mean"] = roc_aucs.mean(axis=1)
            df_cls["roc_auc_std"] = roc_aucs.std(axis=1)            

            spath_cls = os.path.join(self.sdir_for_roc_csv, f"k-fold_mean_std/{cls_str}.csv")
            mngs.io.save(df_cls, spath_cls)
            # dfs_classes.append(df_cls)

    def summarize(
        self,
        n_round=3,
        show=False,
    ):
        """
        1) Take mean and std of scalars/pd.Dataframes for folds.
        2) Replace self.folds_dict with the summarized DataFrames.
        """
        self.summarize_roc()

        _n_folds_all = [
            len(self.folds_dict[k]) for k in self.folds_dict.keys()
        ]  # sometimes includes 0 because AUC curves are not always defined.
        self.n_folds_intended = max(_n_folds_all)

        for i_k, k in enumerate(self.folds_dict.keys()):
            n_folds = _n_folds_all[i_k]

            if n_folds != 0:
                ## listed scalars
                if mngs.general.is_listed_X(self.folds_dict[k], [float, int]):
                    mm = np.mean(self.folds_dict[k])
                    ss = np.std(self.folds_dict[k], ddof=1)
                    sr = pd.DataFrame(
                        data=[mm, ss] + self.folds_dict[k],
                        index=self._mk_cv_index(n_folds),
                        columns=[k],
                    )
                    self.folds_dict[k] = sr.round(n_round)

                ## listed pd.DataFrames
                elif mngs.general.is_listed_X(self.folds_dict[k], pd.DataFrame):
                    zero_df_for_mm = 0 * self.folds_dict[k][0].copy()
                    zero_df_for_ss = 0 * self.folds_dict[k][0].copy()

                    mm = (
                        zero_df_for_mm + np.stack(self.folds_dict[k]).mean(axis=0)
                    ).round(n_round)

                    ss = (
                        zero_df_for_ss
                        + np.stack(self.folds_dict[k]).std(axis=0, ddof=1)
                    ).round(n_round)

                    self.folds_dict[k] = [mm, ss] + [
                        df_fold.round(n_round) for df_fold in self.folds_dict[k]
                    ]

                    if show:
                        print(
                            "\n----------------------------------------\n"
                            f"\n{k}\n"
                            f"\n{n_folds}-fold-CV mean:\n"
                        )
                        pprint(self.folds_dict[k][0])
                        print(f"\n\n{n_folds}-fold-CV std.:\n")
                        pprint(self.folds_dict[k][1])
                        print("\n\n----------------------------------------\n")

                ## listed figures
                elif mngs.general.is_listed_X(
                    self.folds_dict[k], matplotlib.figure.Figure
                ):
                    pass

                else:
                    print(f"{k} was not summarized")
                    print(type(self.folds_dict[k][0]))

    def save(
        self,
        files_to_reproduce=None,
        meta_dict=None,
    ):
        """
        1) Saves the content of self.folds_dict.
        2) Plots the colormap of confusion matrices and saves them.
        3) Saves passed meta_dict under self.sdir

        Example:
            meta_df_1 = pd.DataFrame(data=np.random.rand(3,3))
            meta_dict_1 = {"a": 0}
            meta_dict_2 = {"b": 0}
            meta_dict = {"meta_1.csv": meta_df_1,
                         "meta_1.yaml": meta_dict_1,
                         "meta_2.yaml": meta_dict_1,
            }

        """
        if meta_dict is not None:
            for k, v in meta_dict.items():
                mngs.io.save(v, self.sdir + k)

        for k in self.folds_dict.keys():

            ## pd.Series / pd.DataFrame
            if isinstance(self.folds_dict[k], pd.Series) or isinstance(
                self.folds_dict[k], pd.DataFrame
            ):
                mngs.io.save(self.folds_dict[k], self.sdir + f"{k}.csv")

            ## listed pd.DataFrame
            elif mngs.general.is_listed_X(self.folds_dict[k], pd.DataFrame):
                mngs.io.save(
                    self.folds_dict[k],
                    self.sdir + f"{k}.csv",
                    # indi_suffix=self.cv_index,
                    indi_suffix=self._mk_cv_index(len(self.folds_dict[k])),
                )

            ## listed figures
            elif mngs.general.is_listed_X(self.folds_dict[k], matplotlib.figure.Figure):
                for i_fold, fig in enumerate(self.folds_dict[k]):
                    mngs.io.save(
                        self.folds_dict[k][i_fold], self.sdir + f"{k}/fold#{i_fold}.png"
                    )

            else:
                print(f"{k} was not saved")
                print(type(self.folds_dict[k]))

        if files_to_reproduce is not None:
            if isinstance(files_to_reproduce, list):
                files_to_reproduce = [files_to_reproduce]
            for f in files_to_reproduce:
                mngs.io.save(f, self.sdir)

    def plot_and_save_conf_mats(
        self,
        plt,
        extend_ratio=1.0,
        colorbar=True,
        confmat_plt_config=None,
        sci_notation_kwargs=None,
    ):
        def _inner_plot_conf_mat(
            plt, cm_df, title, extend_ratio=1.0, colorbar=True, sci_notation_kwargs=None
        ):
            labels = list(cm_df.columns)
            fig_conf_mat = mngs.ml.plt.confusion_matrix(
                plt,
                cm_df.T,
                labels=labels,
                title=title,
                x_extend_ratio=extend_ratio,
                y_extend_ratio=extend_ratio,
                colorbar=colorbar,
            )

            if sci_notation_kwargs is not None:
                fig_conf_mat.axes[-1] = mngs.plt.ax_scientific_notation(
                    fig_conf_mat.axes[-1], **sci_notation_kwargs
                )
            return fig_conf_mat

        ## Configures mpl
        mngs.plt.configure_mpl(
            plt,
            **confmat_plt_config,
        )

        ########################################
        ## Prepares confmats dfs
        ########################################
        ## Drops mean and std for the folds
        try:
            conf_mats = self.folds_dict["conf_mat/conf_mat"][-self.n_folds_intended :]

        except Exception as e:
            print(e)
            conf_mats = self.folds_dict["conf_mat/conf_mat"]

        ## Prepaires conf_mat_overall_sum
        conf_mat_zero = 0 * conf_mats[0].copy()  # get the table format
        conf_mat_overall_sum = conf_mat_zero + np.stack(conf_mats).sum(axis=0)

        ########################################
        ## Plots & Saves
        ########################################
        # each fold's conf
        for i_fold, cm in enumerate(conf_mats):
            title = f"Test fold#{i_fold}"
            fig_conf_mat_fold = _inner_plot_conf_mat(
                plt,
                cm,
                title,
                extend_ratio=extend_ratio,
                colorbar=colorbar,
                sci_notation_kwargs=sci_notation_kwargs,
            )
            mngs.io.save(
                fig_conf_mat_fold, self.sdir + f"conf_mat/figs/fold#{i_fold}.png"
            )
            plt.close()

        ## overall_sum conf_mat
        title = f"{self.n_folds_intended}-CV overall sum"
        fig_conf_mat_overall_sum = _inner_plot_conf_mat(
            plt,
            conf_mat_overall_sum,
            title,
            extend_ratio=extend_ratio,
            colorbar=colorbar,
            sci_notation_kwargs=sci_notation_kwargs,
        )
        mngs.io.save(
            fig_conf_mat_overall_sum,
            self.sdir
            + f"conf_mat/figs/{self.n_folds_intended}-fold_cv_overall-sum.png",
        )
        plt.close()


if __name__ == "__main__":
    import random
    import sys

    import mngs
    import numpy as np
    from catboost import CatBoostClassifier, Pool
    from sklearn.datasets import load_digits
    from sklearn.model_selection import StratifiedKFold

    ################################################################################
    ## Sets tee
    ################################################################################
    sdir = mngs.io.mk_spath("./tmp/sdir-ClassificationReporter/")  # "/tmp/sdir/"
    sys.stdout, sys.stderr = mngs.general.tee(sys, sdir)

    ################################################################################
    ## Fixes seeds
    ################################################################################
    mngs.general.fix_seeds(np=np)

    ## Loads
    mnist = load_digits()
    X, T = mnist.data, mnist.target
    labels = mnist.target_names.astype(str)

    ## Main
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    # reporter = ClassificationReporter(sdir)
    mreporter = MultiClassificationReporter(sdir, tgts=["Test1", "Test2"])
    for i_fold, (indi_tra, indi_tes) in enumerate(skf.split(X, T)):
        X_tra, T_tra = X[indi_tra], T[indi_tra]
        X_tes, T_tes = X[indi_tes], T[indi_tes]

        clf = CatBoostClassifier(verbose=False)

        clf.fit(X_tra, T_tra, verbose=False)

        ## Prediction
        pred_proba_tes = clf.predict_proba(X_tes)
        pred_cls_tes = np.argmax(pred_proba_tes, axis=1)

        pred_cls_tes[pred_cls_tes == 9] = 8  # overide 9 as 8 # fixme

        ##############################
        ## Manually adds objects to reporter to save
        ##############################
        ## Figure
        fig, ax = plt.subplots()
        ax.plot(np.arange(10))
        # reporter.add("manu_figs", fig)
        mreporter.add("manu_figs", fig, tgt="Test1")
        mreporter.add("manu_figs", fig, tgt="Test2")

        ## DataFrame
        df = pd.DataFrame(np.random.rand(5, 3))
        # reporter.add("manu_dfs", df)
        mreporter.add("manu_dfs", df, tgt="Test1")
        mreporter.add("manu_dfs", df, tgt="Test2")

        ## Scalar
        scalar = random.random()
        # reporter.add(
        #     "manu_scalars",
        #     scalar,
        # )
        mreporter.add("manu_scalars", scalar, tgt="Test1")
        mreporter.add("manu_scalars", scalar, tgt="Test2")

        ########################################
        ## Metrics
        ########################################
        mreporter.calc_metrics(
            T_tes,
            pred_cls_tes,
            pred_proba_tes,
            labels=labels,
            i_fold=i_fold,
            tgt="Test1",
        )
        mreporter.calc_metrics(
            T_tes,
            pred_cls_tes,
            pred_proba_tes,
            labels=labels,
            i_fold=i_fold,
            tgt="Test2",
        )

    # reporter.summarize(show=True)
    mreporter.summarize(show=True, tgt="Test1")
    mreporter.summarize(show=True, tgt="Test2")

    fake_fpaths = ["fake_file_1.txt", "fake_file_2.txt"]
    for ff in fake_fpaths:
        mngs.io.touch(ff)

    files_to_reproduce = [
        mngs.general.get_this_fpath(when_ipython="/dev/null"),
        *fake_fpaths,
    ]
    # reporter.save(files_to_reproduce=files_to_reproduce)
    mreporter.save(files_to_reproduce=files_to_reproduce, tgt="Test1")
    mreporter.save(files_to_reproduce=files_to_reproduce, tgt="Test2")

    confmat_plt_config = dict(
        figsize=(8, 8),
        # labelsize=8,
        # fontsize=6,
        # legendfontsize=6,
        figscale=2,
        tick_size=0.8,
        tick_width=0.2,
    )

    sci_notation_kwargs = dict(
        order=1,
        fformat="%1.0d",
        scilimits=(-3, 3),
        x=False,
        y=True,
    )  # "%3.1f"

    # sci_notation_kwargs = None
    # reporter.plot_and_save_conf_mats(
    #     plt,
    #     extend_ratio=1.0,
    #     confmat_plt_config=confmat_plt_config,
    #     sci_notation_kwargs=sci_notation_kwargs,
    # )

    mreporter.plot_and_save_conf_mats(
        plt,
        extend_ratio=1.0,
        confmat_plt_config=confmat_plt_config,
        sci_notation_kwargs=sci_notation_kwargs,
        tgt="Test1",
    )
    mreporter.plot_and_save_conf_mats(
        plt,
        extend_ratio=1.0,
        confmat_plt_config=confmat_plt_config,
        sci_notation_kwargs=sci_notation_kwargs,
        tgt="Test2",
    )

    ## EOF
