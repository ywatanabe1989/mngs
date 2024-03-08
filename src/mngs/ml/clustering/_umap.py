#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-03-03 02:12:25 (ywatanabe)"

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import umap.umap_ as umap_orig
from sklearn.preprocessing import LabelEncoder


def umap(
    data_all,
    labels_all,
    axes_titles=None,
    supervised=False,
    title="UMAP Clustering",
    alpha=0.1,
    s=3,
):

    assert len(data_all) == len(labels_all)

    if isinstance(data_all, list):
        data_all = list(data_all)
        labels_all = list(labels_all)

    le = LabelEncoder()

    labels_all_orig = [np.array(labels) for labels in labels_all]

    le.fit(np.hstack(labels_all))
    labels_all = [le.transform(labels) for labels in labels_all]

    umap_model = umap_orig.UMAP(random_state=42)

    if supervised:
        _umap = umap_model.fit(data_all[0], y=labels_all[0])
        title = f"(Supervised) {title}"
    else:
        _umap = umap_model.fit(data_all[0])
        title = f"(Unsupervised) {title}"

    fig, axes = plt.subplots(ncols=len(data_all) + 1, sharex=True, sharey=True)

    for ii, (data, labels) in enumerate(zip(data_all, labels_all)):
        embedding = _umap.transform(data)

        ax = axes[ii + 1]

        axes[0].set_title("Superimposed")
        axes[0].set_aspect("equal")
        palette = "viridis"

        sns.scatterplot(
            x=embedding[:, 0],
            y=embedding[:, 1],
            hue=le.inverse_transform(labels),
            ax=axes[0],
            palette=palette,
            legend="full" if ii == 0 else False,
            s=s,
            alpha=alpha,
        )

        sns.scatterplot(
            x=embedding[:, 0],
            y=embedding[:, 1],
            hue=le.inverse_transform(labels),
            ax=ax,
            palette=palette,
            s=s,
            alpha=alpha,
        )
        ax.set_aspect("equal")

        if axes_titles is not None:
            ax.set_title(axes_titles[ii])

    legend_figs = []
    for i, ax in enumerate(axes):
        legend = ax.get_legend()
        if legend:
            legend_fig = plt.figure(figsize=(3, 2))
            new_legend = legend_fig.gca().legend(
                handles=legend.legendHandles, labels=legend.texts, loc="center"
            )
            legend_fig.canvas.draw()
            legend_filename = f"legend_{i}.png"
            legend_fig.savefig(legend_filename, bbox_inches="tight")
            legend_figs.append(legend_fig)
            plt.close(legend_fig)

    for ax in axes:
        ax.legend_ = None
        # ax.remove_legend()

    fig.suptitle(title)
    fig.supxlabel("UMAP 1")
    fig.supylabel("UMAP 2")

    return fig, legend_figs, _umap


# def umap(
#     data_all,
#     labels_all,
#     axes_titles=None,
#     supervised=False,
#     title="UMAP Clustering",
#     alpha=0.1,
#     s=3,
# ):
#     """
#     Performs UMAP clustering on the given data and labels, and generates a plot with the results.

#     Parameters:
#     - data_all (list of array-like): List of datasets to be used for UMAP clustering.
#     - labels_all (list of array-like): List of label arrays corresponding to the datasets.
#     - axes_titles (list of str, optional): Titles for each subplot axis.
#     - supervised (bool, optional): Whether to use supervised dimensionality reduction. Defaults to False.
#     - title (str, optional): Title for the entire plot. Defaults to "UMAP Clustering".
#     - alpha (float, optional): Alpha value for the scatter plot points. Defaults to 0.1.
#     - s (int, optional): Size of the scatter plot points. Defaults to 3.

#     Returns:
#     - fig (matplotlib.figure.Figure): The main figure object containing the UMAP plots.
#     - legend_figs (list of matplotlib.figure.Figure): List of figures containing the legends.
#     - _umap (umap.UMAP): The fitted UMAP model.
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import seaborn as sns
#     import umap.umap_ as umap_orig
#     from sklearn.preprocessing import LabelEncoder

#     assert len(data_all) == len(labels_all)

#     if isinstance(data_all, list):
#         data_all = list(data_all)
#         labels_all = list(labels_all)

#     le = LabelEncoder()

#     labels_all_orig = [np.array(labels) for labels in labels_all]
#     if labels_all is not None:
#         labels_all = [le.fit_transform(labels) for labels in labels_all]

#     umap_model = umap_orig.UMAP(random_state=42)

#     if supervised:
#         _umap = umap_model.fit(data_all[0], y=labels_all[0])
#         title = f"(Supervised) {title}"
#     else:
#         _umap = umap_model.fit(data_all[0])
#         title = f"(Unsupervised) {title}"

#     fig, axes = plt.subplots(ncols=len(data_all) + 1, sharex=True, sharey=True)

#     for ii, (data, labels) in enumerate(zip(data_all, labels_all)):
#         embedding = _umap.transform(data)
#         ax = axes[ii + 1]

#         axes[0].set_title("Superimposed")
#         axes[0].set_aspect("equal")
#         palette = "viridis"

#         sns.scatterplot(
#             x=embedding[:, 0],
#             y=embedding[:, 1],
#             hue=le.inverse_transform(labels),
#             ax=axes[0],
#             palette=palette,
#             legend="full" if ii == 0 else False,
#             s=s,
#             alpha=alpha,
#         )

#         sns.scatterplot(
#             x=embedding[:, 0],
#             y=embedding[:, 1],
#             hue=le.inverse_transform(labels),
#             ax=ax,
#             palette=palette,
#             s=s,
#             alpha=alpha,
#         )
#         ax.set_aspect("equal")

#         if axes_titles is not None:
#             ax.set_title(axes_titles[ii])

#     legend_figs = []
#     for i, ax in enumerate(axes):
#         legend = ax.get_legend()
#         if legend:
#             legend_fig = plt.figure(figsize=(3, 2))
#             new_legend = legend_fig.gca().add_artist(legend)  # [REVISED]
#             legend.set_bbox_to_anchor((0, 0, 1, 1))
#             legend_fig.canvas.draw()
#             legend_filename = f"legend_{i}.png"
#             legend_fig.savefig(legend_filename, bbox_inches="tight")
#             legend_figs.append(legend_fig)
#             plt.close(legend_fig)

#     fig.suptitle(title)
#     fig.supxlabel("UMAP 1")
#     fig.supylabel("UMAP 2")

#     return fig, legend_figs, _umap


# # def umap(
# #     data_all,
# #     labels_all,
# #     axes_titles=None,
# #     supervised=False,
# #     title="UMAP Clustering",
# #     alpha=0.1,
# #     s=3,
# # ):

# #     assert len(data_all) == len(labels_all)

# #     if isinstance(data_all, list):
# #         data_all = list(data_all)
# #         labels_all = list(labels_all)

# #     le = LabelEncoder()

# #     # Store original labels
# #     labels_all_orig = [np.array(labels) for labels in labels_all]
# #     if labels_all is not None:
# #         labels_all = [le.fit_transform(labels) for labels in labels_all]

# #     umap_model = umap_orig.UMAP(random_state=42)

# #     # Process the primary dataset
# #     if supervised:
# #         _umap = umap_model.fit(data_all[0], y=labels_all[0])
# #         title = f"(Supervised) {title}"
# #     else:
# #         _umap = umap_model.fit(data_all[0])
# #         title = f"(Unsupervised) {title}"

# #     fig, axes = plt.subplots(ncols=len(data_all) + 1, sharex=True, sharey=True)

# #     for ii, (data, labels) in enumerate(zip(data_all, labels_all)):
# #         embedding = _umap.transform(data)
# #         ax = axes[ii + 1]

# #         # Superimposed
# #         axes[0].set_title("Superimposed")
# #         axes[0].set_aspect("equal")
# #         palette = "viridis"

# #         sns.scatterplot(
# #             x=embedding[:, 0],
# #             y=embedding[:, 1],
# #             hue=le.inverse_transform(labels),
# #             ax=axes[0],
# #             palette=palette,
# #             legend="full" if ii == 0 else False,
# #             s=s,
# #             alpha=alpha,
# #         )

# #         # Each data
# #         sns.scatterplot(
# #             x=embedding[:, 0],
# #             y=embedding[:, 1],
# #             hue=le.inverse_transform(labels),
# #             ax=ax,
# #             palette=palette,
# #             s=s,
# #             alpha=alpha,
# #         )
# #         ax.set_aspect("equal")

# #         if axes_titles is not None:
# #             ax.set_title(axes_titles[ii])

# #     # Save legends as separate figures and store them in a list
# #     legend_figs = []
# #     for i, ax in enumerate(axes):
# #         # Extract the legend from the current axis
# #         legend = ax.get_legend()
# #         if legend:
# #             # Create a new figure for the legend
# #             legend_fig = plt.figure(figsize=(3, 2))
# #             new_legend = legend_fig._get_axes().add_artist(legend)
# #             legend.set_bbox_to_anchor((0, 0, 1, 1))
# #             legend_fig.canvas.draw()
# #             # Save the legend as a PNG file
# #             legend_filename = f"legend_{i}.png"
# #             legend_fig.savefig(legend_filename, bbox_inches="tight")
# #             # Store the legend figure in the list
# #             legend_figs.append(legend_fig)
# #             plt.close(legend_fig)  # Close the figure to free memory

# #     fig.suptitle(title)
# #     fig.supxlabel("UMAP 1")
# #     fig.supylabel("UMAP 2")

# #     # Return the main figure, the list of legend figures, and the UMAP model
# #     return fig, legend_figs, _umap


# # # from itertools import cycle

# # # import matplotlib.pyplot as plt
# # # import mngs
# # # import numpy as np
# # # import seaborn as sns
# # # import umap.umap_ as umap_orig
# # # from sklearn.preprocessing import LabelEncoder


# # # def umap(
# # #     data_all,
# # #     labels_all,
# # #     axes_titles=None,
# # #     supervised=False,
# # #     title="UMAP Clustering",
# # #     alpha=0.1,
# # #     s=3,
# # #     # colors=None,
# # # ):

# # #     assert len(data_all) == len(labels_all)

# # #     if isinstance(data_all, list):
# # #         data_all = list(data_all)
# # #         labels_all = list(labels_all)

# # #     le = mngs.ml.utils.LabelEncoder()
# # #     # le = LabelEncoder()

# # #     # Store original labels
# # #     labels_all_orig = [np.array(labels) for labels in labels_all]
# # #     if labels_all is not None:
# # #         # labels_uq = np.unique(np.hstack(labels_all))
# # #         # le.fit(labels_uq)
# # #         labels_all = [le.fit_transform(labels) for labels in labels_all]

# # #     umap_model = umap_orig.UMAP(random_state=42)

# # #     # Process the primary dataset
# # #     if supervised:
# # #         _umap = umap_model.fit(data_all[0], y=labels_all[0])
# # #         title = f"(Supervised) {title}"
# # #     else:
# # #         _umap = umap_model.fit(data_all[0])
# # #         title = f"(Unsupervised) {title}"

# # #     fig, axes = plt.subplots(ncols=len(data_all) + 1, sharex=True, sharey=True)
# # #     # # Create a color palette that maps each unique label to a color
# # #     # unique_labels = np.unique(np.hstack(labels_all_orig))
# # #     # if colors is not None:
# # #     #     palette = dict(zip(unique_labels, cycle(colors)))
# # #     # else:
# # #     #     palette = sns.color_palette("hsv", len(unique_labels))
# # #     #     palette = dict(zip(unique_labels, palette))

# # #     # if colors is not None:
# # #     #     color_cycle = cycle(colors)
# # #     # else:
# # #     #     color_cycle = None

# # #     # for ii, (data, labels, labels_orig) in enumerate(
# # #     #     zip(data_all, labels_all, labels_all_orig)
# # #     # ):
# # #     for ii, (data, labels) in enumerate(zip(data_all, labels_all)):
# # #         embedding = _umap.transform(data)
# # #         ax = axes[ii + 1]

# # #         # Superimposed
# # #         axes[0].set_title("Superimposed")
# # #         axes[0].set_aspect("equal")
# # #         palette = "viridis"
# # #         # if color_cycle:
# # #         #     palette = sns.color_palette(
# # #         #         [next(color_cycle) for _ in range(len(np.unique(labels)))]
# # #         #     )
# # #         # else:
# # #         #     palette = "viridis"

# # #         sns.scatterplot(
# # #             x=embedding[:, 0],
# # #             y=embedding[:, 1],
# # #             hue=le.inverse_transform(labels),
# # #             ax=axes[0],
# # #             palette=palette,
# # #             legend="full" if ii == 0 else False,
# # #             s=s,
# # #             alpha=alpha,
# # #         )

# # #         # Each data
# # #         sns.scatterplot(
# # #             x=embedding[:, 0],
# # #             y=embedding[:, 1],
# # #             hue=le.inverse_transform(labels),
# # #             ax=ax,
# # #             palette=palette,
# # #             s=s,
# # #             alpha=alpha,
# # #         )
# # #         ax.set_aspect("equal")

# # #         if axes_titles is not None:
# # #             ax.set_title(axes_titles[ii])

# # #     # Remove the legends from the individual axes
# # #     for ax in axes:
# # #         ax.legend(loc="upper left")
# # #         legend = ax.get_legend()
# # #         if legend:
# # #             legend_fig = plt.figure(figsize=(3, 2))


# # #     fig.suptitle(title)
# # #     fig.supxlabel("UMAP 1")
# # #     fig.supylabel("UMAP 2")

# # #     return fig, _umap
