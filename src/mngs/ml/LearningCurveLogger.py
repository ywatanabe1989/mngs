#!/usr/bin/env python3

from pprint import pprint

# import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import mngs


class LearningCurveLogger(object):
    def __init__(
        self,
    ):
        self.steps_str = ("Training", "Test")
        self.logged_dict = mngs.general.listed_dict(self.steps_str)
        for k in self.steps_str:
            self.logged_dict[k] = {}

    def __call__(self, log_dict, step="Training"):
        for k in log_dict.keys():
            try:
                self.logged_dict[step][k].append(log_dict[k])
            except:
                self.logged_dict[step][k] = []
                self.logged_dict[step][k].append(log_dict[k])

    def to_dfs(self, steps_str=["Training", "Test"]):  # fixme
        self.dfs = {}
        self.dfs_pivot = {}
        for step in steps_str:
            logged_dict = self.logged_dict[step]
            length = len(logged_dict["i_epoch"])
            df = pd.DataFrame(
                {
                    "step": [step for _ in range(length)],
                    "i_mouse_test": logged_dict["i_mouse_test"],
                    "i_epoch": logged_dict["i_epoch"],
                    "i_global": logged_dict["i_global"],
                    "loss": logged_dict["loss"],
                    "Balanced ACC": logged_dict["balanced_ACC"],
                    "pred_proba": [logged_dict["pred_proba"][i] for i in range(length)],
                    "gt_label": [logged_dict["gt_label"][i] for i in range(length)],
                }
            )  # fixme

            self.dfs[step] = df

            # pivot version to plot
            self.dfs_pivot[step] = (
                self.dfs[step][
                    [
                        "i_global",
                        "Balanced ACC",
                        "loss",
                    ]
                ]
                .pivot_table(columns=["i_global"], aggfunc="mean")
                .T
            )

    def print_learning_curves_in_digits(self, step="Training"):
        print(step)

        try:
            df = pd.DataFrame(
                {
                    "i_epoch": self.logged_dict[step]["i_epoch"],
                    "Balanced ACC": self.logged_dict[step]["balanced_ACC"],
                }
            ).pivot_table(columns=["i_epoch"], aggfunc="mean")
            pprint(df)
        except:
            pass

        print()

    def plot_learning_curves(
        self,
        plt,
        i_mouse_test=None,
        max_epochs=None,
        window_size_sec=None,
        max_n_ticks=4,
    ):

        self.to_dfs()

        ## Plot
        fig, ax = plt.subplots(2, 1, sharex=True, sharey=False)
        title = "fold#{}; MAX EPOCHS = {}; WINDOW SIZE = {} [sec]".format(
            i_mouse_test, max_epochs, window_size_sec
        )
        fig.text(0.5, 0.95, title, ha="center")

        keys_to_plot = ["loss", "Balanced ACC"]

        COLOR_DICT = {
            "Training": "blue",
            "Test": "green",
        }

        for i_plt, plt_k in enumerate(keys_to_plot):
            for step_k in self.dfs_pivot.keys():
                if step_k == "Training":  # line
                    ax[i_plt].plot(
                        self.dfs_pivot[step_k].index,
                        self.dfs_pivot[step_k][plt_k],
                        label=step_k,
                        color=COLOR_DICT[step_k],
                        linewidth=1,
                        # linewidth=3,
                    )

                if step_k == "Test":  # scatter
                    ax[i_plt].scatter(
                        self.dfs_pivot[step_k].index,
                        self.dfs_pivot[step_k][plt_k],
                        label=step_k,
                        color=COLOR_DICT[step_k],
                        # s=150,
                        s=50,
                        alpha=0.9,
                    )

                if "Balanced ACC" in plt_k:
                    ax[i_plt].set_ylim(0, 1)

        ax[0].legend()
        ax[1].legend()

        ax[1].set_xlabel("Iteration#")

        ax[0].set_ylabel("Loss")
        ax[1].set_ylabel("Balanced Accuracy")

        for a in ax:
            a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(max_n_ticks))
            a.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(max_n_ticks))

        return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch
    from scipy.special import softmax

    lc_logger = LearningCurveLogger()
    i_global = 0
    batch_size = 64
    n_classes = 2
    i_mouse_test = 0
    max_epochs = 3

    for i_epoch in range(max_epochs):
        step = "Training"
        for i_batch in enumerate(range(1000)):

            log_dict = {
                "loss": float(np.random.rand(1)),
                "balanced_ACC": float(np.random.rand(1)),
                "pred_proba": softmax(np.random.rand(batch_size, n_classes), axis=-1),
                "gt_label": np.random.randint(n_classes, size=batch_size),
                "i_mouse_test": i_mouse_test,
                "i_epoch": i_epoch,
                "i_global": i_global,
            }

            lc_logger(log_dict, step=step)

            i_global += 1

    step = "Test"
    for i_batch in enumerate(range(1000)):

        log_dict = {
            "loss": float(np.random.rand(1)),
            "balanced_ACC": float(np.random.rand(1)),
            "pred_proba": softmax(np.random.rand(batch_size, n_classes), axis=-1),
            "gt_label": np.random.randint(n_classes, size=batch_size),
            "i_mouse_test": i_mouse_test,
            "i_epoch": i_epoch,
            "i_global": i_global,
        }

        lc_logger(log_dict, step=step)

    mngs.plt.configure_mpl(
        plt,
        figsize=(8.7, 10),
        labelsize=8,
        fontsize=7,
        legendfontsize=6,
        tick_size=0.8,
        tick_width=0.2,
    )

    fig = lc_logger.plot_learning_curves(
        plt,
        i_mouse_test=i_mouse_test,
        max_epochs=max_epochs,
        window_size_sec=1024,
    )
    fig.show()
