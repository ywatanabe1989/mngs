#!/usr/bin/env python3
import matplotlib
import mngs
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix


def conf_mat(
    plt,
    cm=None,
    y_true=None,
    y_pred=None,
    y_pred_proba=None,
    labels=None,
    sorted_labels=None,
    pred_labels=None,
    sorted_pred_labels=None,
    true_labels=None,
    sorted_true_labels=None,
    label_rotation_xy=(15, 15),
    title=None,
    colorbar=True,
    x_extend_ratio=1.0,
    y_extend_ratio=1.0,
    spath=None,
):
    """
    Inverse the y-axis and plot the confusion matrix as a heatmap.
    The predicted labels (in x-axis) is symboled with hat (^).
    The plt object is passed to adjust the figure size

    cm = sklearn.metrics.confusion_matrix(y_test, y_pred)


    cm = np.random.randint(low=0, high=10, size=[3,4])
    x: predicted labels
    y: true_labels


    kwargs:

        "extend_ratio":
            Determines how much the axes objects (not the fig object) are extended
            in the vertical direction.

    """

    if (y_pred_proba is not None) and (y_pred is None):
        y_pred = y_pred_proba.argmax(axis=-1)

    assert (cm is not None) or ((y_true is not None) and (y_pred is not None))

    if not cm:
        cm = sklearn_confusion_matrix(y_true, y_pred)

    # Dataframe
    df = pd.DataFrame(data=cm).copy()

    # To LaTeX styles
    if pred_labels is not None:
        pred_labels = [mngs.general.to_the_latex_style(l) for l in pred_labels]
    if true_labels is not None:
        true_labels = [mngs.general.to_the_latex_style(l) for l in true_labels]
    if labels is not None:
        labels = [mngs.general.to_the_latex_style(l) for l in labels]
    if sorted_labels is not None:
        sorted_labels = [
            mngs.general.to_the_latex_style(l) for l in sorted_labels
        ]

    # Prediction Labels: columns
    if pred_labels is not None:
        df.columns = pred_labels
    elif (pred_labels is None) and (labels is not None):
        df.columns = labels

    # Ground Truth Labels: index
    if true_labels is not None:
        df.index = true_labels
    elif (true_labels is None) and (labels is not None):
        df.index = labels

    # Sort based on sorted_labels here
    if sorted_labels is not None:
        assert set(sorted_labels) == set(labels)
        df = df.reindex(index=sorted_labels, columns=sorted_labels)

    # Main
    fig, ax = plt.subplots()
    res = sns.heatmap(
        df,
        annot=True,
        annot_kws={"size": plt.rcParams["font.size"]},
        fmt=".0f",
        cmap="Blues",
        cbar=False,
    )  # Here, don't plot color bar.

    # Adds comma separator for the annotated int texts
    for t in ax.texts:
        t.set_text("{:,d}".format(int(t.get_text())))

    # Inverts the y-axis
    res.invert_yaxis()

    # Makes the frame visible
    for _, spine in res.spines.items():
        spine.set_visible(False)

    # Labels
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    # Appearances
    ax = mngs.plt.ax.extend(ax, x_extend_ratio, y_extend_ratio)
    if df.shape[0] == df.shape[1]:
        ax.set_box_aspect(1)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=label_rotation_xy[0],
        fontdict={"verticalalignment": "top"},
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=label_rotation_xy[1],
        fontdict={"horizontalalignment": "right"},
    )
    # The size
    bbox = ax.get_position()
    left_orig = bbox.x0
    width_orig = bbox.x1 - bbox.x0
    g_x_orig = left_orig + width_orig / 2.0
    width_tgt = width_orig * x_extend_ratio  # x_extend_ratio
    dx = width_orig - width_tgt

    # Adjusts the sizes of the confusion matrix and colorbar
    if colorbar == True:  # fixme
        divider = make_axes_locatable(ax)  # Gets region from the ax
        cax = divider.append_axes("right", size="5%", pad=0.1)
        # cax = divider.new_horizontal(size="5%", pad=1, pack_start=True)
        cax = mngs.plt.ax.set_pos(fig, cax, -dx * 2.54, 0)
        fig.add_axes(cax)

        """
        axpos = ax.get_position()
        caxpos = cax.get_position()

        AddAxesBBoxRect(fig, ax, ec="r")
        AddAxesBBoxRect(fig, cax, ec="b")

        fig.text(
            axpos.x0 + 0.01, axpos.y0 + 0.01, "after colorbar", weight="bold", color="r"
        )

        fig.text(
            caxpos.x1 + 0.01,
            caxpos.y1 - 0.01,
            "cax position",
            va="top",
            weight="bold",
            color="b",
            rotation="vertical",
        )
        """

        # Plots colorbar and adjusts the size
        vmax = np.array(df).max().astype(int)
        norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap="Blues"),
            cax=cax,
            # shrink=0.68,
        )
        cbar.locator = ticker.MaxNLocator(nbins=4)  # tick_locator
        cbar.update_ticks()
        # cbar.outline.set_edgecolor("#f9f2d7")
        cbar.outline.set_edgecolor("white")

    if spath is not None:
        mngs.io.save(fig, spath)

    return fig, cm


# def AddAxesBBoxRect(fig, ax, ec="k"):
#     from matplotlib.patches import Rectangle

#     axpos = ax.get_position()
#     rect = fig.patches.append(
#         Rectangle(
#             (axpos.x0, axpos.y0),
#             axpos.width,
#             axpos.height,
#             ls="solid",
#             lw=2,
#             ec=ec,
#             fill=False,
#             transform=fig.transFigure,
#         )
#     )
#     return rect


if __name__ == "__main__":

    import mngs

    y_true, y_pred = mngs.io.load("/tmp/tmp.pkl")

    fig, cm = conf_mat(plt, y_true=y_true, y_pred=y_pred)

    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    import sys

    import matplotlib.pyplot as plt
    import numpy as np
    import sklearn
    from sklearn import datasets, svm

    # from sklearn.metrics import plot_confusion_matrix
    from sklearn.model_selection import train_test_split

    sys.path.append(".")
    import mngs

    # Imports some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    class_names = iris.target_names

    # Splits the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Runs classifier, using a model that is too regularized (C too low) to see
    # the impact on the results
    classifier = svm.SVC(kernel="linear", C=0.01).fit(X_train, y_train)

    ## Checks the confusion_matrix function
    y_pred = classifier.predict(X_test)
    cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
    cm **= 3

    cm = np.random.randint(low=0, high=10, size=[3, 4])

    mngs.plt.configure_mpl(
        plt,
        # figsize=(4, 8),
        # figsize=(4, 8),
        # fontsize=6,
        # labelsize=8,
        # legendfontsize=7,
        # tick_size=0.8,
        # tick_width=0.2,
    )

    # labels = class_names
    pred_labels = ["A", "B", "C"]
    true_labels = ["a", "b", "c", "d"]

    fig, cm = conf_mat(
        plt,
        cm,
        # labels=class_names,
        pred_labels=pred_labels,
        true_labels=true_labels,
        label_rotation_xy=(60, 60),
        x_extend_ratio=1.0,
        colorbar=True,
    )

    fig.axes[-1] = mngs.plt.ax.sci_note(
        fig.axes[-1],
        3,
        fformat="%3.1f",
        # fformat="%d",
        y=True,
    )

    fig.show()

    ## EOF
