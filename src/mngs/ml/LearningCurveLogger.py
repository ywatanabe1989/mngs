#!/usr/bin/env python3

import re
from collections import defaultdict
from pprint import pprint

import matplotlib
import mngs
import pandas as pd


class LearningCurveLogger(object):
    def __init__(
        self,
    ):
        self.logged_dict = defaultdict(dict)

    def __call__(self, dict_to_log, step):
        """
        dict_to_log | str
            Example:
                dict_to_log = {
                    "loss_plot": float(loss),
                    "balanced_ACC_plot": float(bACC),
                    "pred_proba": pred_proba.detach().cpu().numpy(),
                    "gt_label": T.cpu().numpy(),
                    "i_fold": i_fold,
                    "i_epoch": i_epoch,
                    "i_global": i_global,
                }

        step | str
            "Training", "Validation", or "Test"
        """
        assert step in ["Training", "Validation", "Test"]

        for k in dict_to_log.keys():
            try:
                self.logged_dict[step][k].append(dict_to_log[k])
            except:
                self.logged_dict[step].update({k: []})
                self.logged_dict[step][k].append(dict_to_log[k])

    @staticmethod
    def _rename_if_key_to_plot(keys):
        def _rename_key_to_plot(key_to_plot):
            renamed = key_to_plot[:-5]
            renamed = renamed.replace("_", " ")
            capitalized = []
            for s in renamed.split(" "):
                if not re.search("[A-Z]", s[0]):
                    capitalized.append(s.capitalize())
                else:
                    capitalized.append(s)

            renamed = mngs.general.connect_strs(capitalized, filler=" ")
            return renamed

        if isinstance(keys, str):
            keys = [keys]

        out = []
        for key in keys:
            if key[-5:] == "_plot":
                out.append(_rename_key_to_plot(key))
            else:
                out.append(key)

        if len(out) == 1:
            out = out[0]

        return out

    def plot_learning_curves(
        self,
        plt,
        title=None,
        max_n_ticks=4,
        linewidth=3,
        scattersize=150,
    ):

        self._to_dfs()

        ########################################
        ## Parameters
        ########################################
        COLOR_DICT = {
            "Training": "blue",
            "Validation": "green",
            "Test": "red",
        }

        keys_to_plot = self._finds_keys_to_plot()

        ########################################
        ## Plot
        ########################################
        fig, axes = plt.subplots(len(keys_to_plot), 1, sharex=True, sharey=False)
        axes[-1].set_xlabel("Iteration#")
        fig.text(0.5, 0.95, title, ha="center")

        for i_plt, plt_k in enumerate(keys_to_plot):
            ax = axes[i_plt]
            ax.set_ylabel(self._rename_if_key_to_plot(plt_k))
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(max_n_ticks))
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(max_n_ticks))

            if re.search("[aA][cC][cC]", plt_k):  # acc, ylim, yticks
                ax.set_ylim(0, 1)
                ax.set_yticks([0, 0.5, 1.0])

            for step_k in self.dfs_pivot.keys():

                if step_k == "Training":  # line
                    ax.plot(
                        self.dfs_pivot[step_k].index,
                        self.dfs_pivot[step_k][plt_k],
                        label=step_k,
                        color=mngs.plt.colors.to_RGBA(COLOR_DICT[step_k], alpha=0.9),
                        linewidth=linewidth,
                    )
                    ax.legend()

                    ## Epoch starts ponts
                    epoch_starts = abs(
                        self.dfs_pivot[step_k]["i_epoch"]
                        - self.dfs_pivot[step_k]["i_epoch"].shift(-1)
                    )
                    indi_global_epoch_starts = [0] + list(
                        epoch_starts[epoch_starts == 1].index
                    )

                    for i_epoch, i_global_epoch_start in enumerate(
                        indi_global_epoch_starts
                    ):
                        ax.axvline(
                            x=i_global_epoch_start,
                            ymin=ax.get_ylim()[0],
                            ymax=ax.get_ylim()[1],
                            linestyle="--",
                            color=mngs.plt.colors.to_RGBA("gray", alpha=0.5),
                        )
                        ax.text(
                            i_global_epoch_start,
                            ax.get_ylim()[1] * 1.0,
                            f"epoch#{i_epoch}",
                        )

                if (step_k == "Validation") or (step_k == "Test"):  # scatter
                    ax.scatter(
                        self.dfs_pivot[step_k].index,
                        self.dfs_pivot[step_k][plt_k],
                        label=step_k,
                        color=mngs.plt.colors.to_RGBA(COLOR_DICT[step_k], alpha=0.9),
                        s=scattersize,
                        alpha=0.9,
                    )
                    ax.legend()

        return fig

    def print(self, step):

        df = pd.DataFrame(
            {
                k: self.logged_dict[step][k]
                for k in ["i_epoch"] + self._finds_keys_to_plot()
            }
        ).pivot_table(columns=["i_epoch"], aggfunc="mean")

        df = df.set_index(pd.Series(self._rename_if_key_to_plot(list(df.index))))

        print("\n----------------------------------------\n")
        print(f"\n{step}:\n")
        pprint(df)
        print("\n----------------------------------------\n")

    def _finds_keys_to_plot(self):
        ########################################
        ## finds keys to plot
        ########################################
        _steps_str = list(self.logged_dict.keys())
        _, keys_to_plot = mngs.general.search(
            # ["_plot"],
            "_plot",
            list(self.logged_dict[_steps_str[0]].keys()),
        )
        return keys_to_plot

    def _to_dfs(self):
        self.dfs = {}
        self.dfs_pivot = {}
        for step in self.logged_dict.keys():
            df_s = mngs.general.pandas.force_dataframe(self.logged_dict[step])
            df_s_pvt_on_i_global = df_s.pivot_table(
                columns="i_global", aggfunc="mean"
            ).T

            self.dfs[step] = df_s
            self.dfs_pivot[step] = df_s_pvt_on_i_global


if __name__ == "__main__":
    import warnings

    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    from sklearn.metrics import balanced_accuracy_score
    from torch.utils.data import DataLoader, TensorDataset
    from torch.utils.data.dataset import Subset
    from torchvision import datasets

    import sys

    ################################################################################
    ## Sets tee
    ################################################################################
    sdir = mngs.general.path.mk_spath("")  # "/tmp/sdir/"
    sys.stdout, sys.stderr = mngs.general.tee(sys, sdir)

    ################################################################################
    ## NN
    ################################################################################
    class Perceptron(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(28 * 28, 50)
            self.l2 = nn.Linear(50, 10)

        def forward(self, x):
            x = x.view(-1, 28 * 28)
            x = self.l1(x)
            x = self.l2(x)
            return x

    ################################################################################
    ## Prepaires demo data
    ################################################################################
    ## Downloads
    _ds_tra_val = datasets.MNIST("/tmp/mnist", train=True, download=True)
    _ds_tes = datasets.MNIST("/tmp/mnist", train=False, download=True)

    ## Training-Validation splitting
    n_samples = len(_ds_tra_val)  # n_samples is 60000
    train_size = int(n_samples * 0.8)  # train_size is 48000

    subset1_indices = list(range(0, train_size))  # [0,1,.....47999]
    subset2_indices = list(range(train_size, n_samples))  # [48000,48001,.....59999]

    _ds_tra = Subset(_ds_tra_val, subset1_indices)
    _ds_val = Subset(_ds_tra_val, subset2_indices)

    ## to tensors
    ds_tra = TensorDataset(
        _ds_tra.dataset.data.to(torch.float32),
        _ds_tra.dataset.targets,
    )
    ds_val = TensorDataset(
        _ds_val.dataset.data.to(torch.float32),
        _ds_val.dataset.targets,
    )
    ds_tes = TensorDataset(
        _ds_tes.data.to(torch.float32),
        _ds_tes.targets,
    )

    ## to dataloaders
    batch_size = 64
    dl_tra = DataLoader(
        dataset=ds_tra,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    dl_val = DataLoader(
        dataset=ds_val,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )

    dl_tes = DataLoader(
        dataset=ds_tes,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )

    ################################################################################
    ## Preparation
    ################################################################################
    model = Perceptron()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    softmax = nn.Softmax(dim=-1)

    ################################################################################
    ## Main
    ################################################################################
    lc_logger = LearningCurveLogger()
    i_global = 0

    n_classes = len(dl_tra.dataset.tensors[1].unique())
    i_fold = 0
    max_epochs = 3

    for i_epoch in range(max_epochs):
        step = "Validation"
        for i_batch, batch in enumerate(dl_val):

            X, T = batch
            logits = model(X)
            pred_proba = softmax(logits)
            pred_class = pred_proba.argmax(dim=-1)
            loss = loss_func(logits, T)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                bACC = balanced_accuracy_score(T, pred_class)

            dict_to_log = {
                "loss_plot": float(loss),
                "balanced_ACC_plot": float(bACC),
                "pred_proba": pred_proba.detach().cpu().numpy(),
                "gt_label": T.cpu().numpy(),
                "i_fold": i_fold,
                "i_epoch": i_epoch,
                "i_global": i_global,
            }
            lc_logger(dict_to_log, step)

        lc_logger.print(step)

        step = "Training"
        for i_batch, batch in enumerate(dl_tra):
            optimizer.zero_grad()

            X, T = batch
            logits = model(X)
            pred_proba = softmax(logits)
            pred_class = pred_proba.argmax(dim=-1)
            loss = loss_func(logits, T)

            loss.backward()
            optimizer.step()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                bACC = balanced_accuracy_score(T, pred_class)

            dict_to_log = {
                "loss_plot": float(loss),
                "balanced_ACC_plot": float(bACC),
                "pred_proba": pred_proba.detach().cpu().numpy(),
                "gt_label": T.cpu().numpy(),
                "i_fold": i_fold,
                "i_epoch": i_epoch,
                "i_global": i_global,
            }
            lc_logger(dict_to_log, step)

            i_global += 1

        lc_logger.print(step)

    step = "Test"
    for i_batch, batch in enumerate(dl_tes):

        X, T = batch
        logits = model(X)
        pred_proba = softmax(logits)
        pred_class = pred_proba.argmax(dim=-1)
        loss = loss_func(logits, T)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            bACC = balanced_accuracy_score(T, pred_class)

        dict_to_log = {
            "loss_plot": float(loss),
            "balanced_ACC_plot": float(bACC),
            "pred_proba": pred_proba.detach().cpu().numpy(),
            "gt_label": T.cpu().numpy(),
            "i_fold": i_fold,
            "i_epoch": i_epoch,
            "i_global": i_global,
        }
        lc_logger(dict_to_log, step)

    lc_logger.print(step)

    # mngs.plt.configure_mpl(
    #     plt,
    #     figsize=(8.7, 10),
    #     labelsize=8,
    #     fontsize=7,
    #     legendfontsize=6,
    #     tick_size=0.8,
    #     tick_width=0.2,
    # )

    fig = lc_logger.plot_learning_curves(plt, title=f"fold#{i_fold}")
    # fig.show()
    mngs.general.save(fig, sdir + f"fold#{i_fold}.png")
