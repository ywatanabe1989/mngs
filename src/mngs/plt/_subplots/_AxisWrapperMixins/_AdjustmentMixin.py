#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-13 14:53:49 (ywatanabe)"
# File: ./mngs_repo/src/mngs/plt/_subplots/_AxisWrapperMixins/_AdjustmentMixin.py

from typing import List, Optional, Union

"""
Functionality:
    * Provides methods for adjusting matplotlib plot aesthetics
Input:
    * Label rotations, axis labels, tick values, spines, and plot extent
Output:
    * Modified matplotlib axis object
Prerequisites:
    * matplotlib, mngs.plt.ax
"""

from ....plt import ax as ax_module


class AdjustmentMixin:
    """Mixin class for matplotlib axis adjustments."""

    def rotate_labels(
        self,
        x: float = 30,
        y: float = 30,
        x_ha: str = "right",
        y_ha: str = "center",
    ) -> None:
        self.axis = ax_module.rotate_labels(
            self.axis, x=x, y=y, x_ha=x_ha, y_ha=y_ha
        )

    def legend(self, location: str = "upper left") -> None:
        return self.axis.legend(loc=location)

    def set_xyt(
        self,
        x: Optional[str] = None,
        y: Optional[str] = None,
        tt: Optional[str] = None,
        format_labels: bool = True,
    ) -> None:
        self.axis = ax_module.set_xyt(
            self.axis,
            x=x,
            y=y,
            t=tt,
            format_labels=format_labels,
        )

    def set_supxyt(
        self,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        format_labels: bool = True,
    ) -> None:
        self.axis = ax_module.set_supxyt(
            self.axis,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            format_labels=format_labels,
        )

    def set_ticks(
        self,
        xvals: Optional[List[Union[int, float]]] = None,
        xticks: Optional[List[str]] = None,
        yvals: Optional[List[Union[int, float]]] = None,
        yticks: Optional[List[str]] = None,
    ) -> None:
        self.axis = ax_module.set_ticks(
            self.axis,
            xvals=xvals,
            xticks=xticks,
            yvals=yvals,
            yticks=yticks,
        )

    def set_n_ticks(self, n_xticks: int = 4, n_yticks: int = 4) -> None:
        self.axis = ax_module.set_n_ticks(
            self.axis, n_xticks=n_xticks, n_yticks=n_yticks
        )

    def hide_spines(
        self,
        top: bool = True,
        bottom: bool = True,
        left: bool = True,
        right: bool = True,
        ticks: bool = True,
        labels: bool = True,
    ) -> None:
        self.axis = ax_module.hide_spines(
            self.axis,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            ticks=ticks,
            labels=labels,
        )

    def extend(self, x_ratio: float = 1.0, y_ratio: float = 1.0) -> None:
        self.axis = ax_module.extend(
            self.axis, x_ratio=x_ratio, y_ratio=y_ratio
        )

    def shift(self, dx: float = 0, dy: float = 0) -> None:
        self.axis = ax_module.shift(self.axis, dx=dx, dy=dy)

    # def rotate_labels(self, x=30, y=30, x_ha="right", y_ha="center"):
    #     self.axis = ax_module.rotate_labels(
    #         self.axis, x=x, y=y, x_ha=x_ha, y_ha=y_ha
    #     )

    # def legend(self, loc="upper left"):
    #     return self.axis.legend(loc=loc)

    # def set_xyt(
    #     self,
    #     x=None,
    #     y=None,
    #     t=None,
    #     format_labels=True,
    # ):
    #     self.axis = ax_module.set_xyt(
    #         self.axis,
    #         x=x,
    #         y=y,
    #         t=t,
    #         format_labels=format_labels,
    #     )

    # def set_supxyt(
    #     self,
    #     xlabel=None,
    #     ylabel=None,
    #     title=None,
    #     format_labels=True,
    # ):
    #     self.axis = ax_module.set_supxyt(
    #         self.axis,
    #         xlabel=xlabel,
    #         ylabel=ylabel,
    #         title=title,
    #         format_labels=format_labels,
    #     )

    # def set_ticks(
    #     self,
    #     xvals=None,
    #     xticks=None,
    #     yvals=None,
    #     yticks=None,
    #     **kwargs,
    # ):

    #     self.axis = ax_module.set_ticks(
    #         self.axis,
    #         xvals=xvals,
    #         xticks=xticks,
    #         yvals=yvals,
    #         yticks=yticks,
    #     )

    # def set_n_ticks(self, n_xticks=4, n_yticks=4):
    #     self.axis = ax_module.set_n_ticks(
    #         self.axis, n_xticks=n_xticks, n_yticks=n_yticks
    #     )

    # def hide_spines(
    #     self,
    #     top=True,
    #     bottom=True,
    #     left=True,
    #     right=True,
    #     ticks=True,
    #     labels=True,
    # ):
    #     self.axis = ax_module.hide_spines(
    #         self.axis,
    #         top=top,
    #         bottom=bottom,
    #         left=left,
    #         right=right,
    #         ticks=ticks,
    #         labels=labels,
    #     )

    # def extend(self, x_ratio=1.0, y_ratio=1.0):
    #     self.axis = ax_module.extend(
    #         self.axis, x_ratio=x_ratio, y_ratio=y_ratio
    #     )

    # def shift(self, dx=0, dy=0):
    #     self.axis = ax_module.shift(self.axis, dx=dx, dy=dy)


# EOF
