#!/usr/bin/env python3


def set_pos(
    fig, ax, x_delta_offset_cm, y_delta_offset_cm, dragh=False, dragv=False
):
    """
    Adjusts the position of an Axes object within a Figure by a specified offset in centimeters.

    Parameters:
    - fig (matplotlib.figure.Figure): The Figure object containing the Axes.
    - ax (matplotlib.axes.Axes): The Axes object to modify.
    - x_delta_offset_cm (float): The horizontal offset in centimeters to adjust the Axes position.
    - y_delta_offset_cm (float): The vertical offset in centimeters to adjust the Axes position.
    - dragh (bool): If True, reduces the width of the Axes by the horizontal offset.
    - dragv (bool): If True, reduces the height of the Axes by the vertical offset.

    Returns:
    - ax (matplotlib.axes.Axes): The modified Axes object with the adjusted position.
    """

    bbox = ax.get_position()

    # Calculates delta ratios
    fig_width_inch, fig_height_inch = fig.get_size_inches()

    x_delta_offset_inch = x_delta_offset_cm / 2.54
    y_delta_offset_inch = y_delta_offset_cm / 2.54

    x_delta_offset_ratio = x_delta_offset_inch / fig_width_inch
    y_delta_offset_ratio = y_delta_offset_inch / fig_height_inch  # [REVISED]

    # Determines updated bbox position
    left = bbox.x0 + x_delta_offset_ratio
    bottom = bbox.y0 + y_delta_offset_ratio
    width = bbox.width  # [REVISED]
    height = bbox.height  # [REVISED]

    if dragh:
        width -= x_delta_offset_ratio

    if dragv:
        height -= y_delta_offset_ratio

    ax.set_position([left, bottom, width, height])  # [REVISED]

    return ax


# def set_pos(
#     fig,
#     ax,
#     x_delta_offset_cm,
#     y_delta_offset_cm,
#     dragh=False,
#     dragv=False,
# ):

#     bbox = ax.get_position()

#     ## Calculates delta ratios
#     fig_width_inch, fig_height_inch = fig.get_size_inches()

#     x_delta_offset_inch = float(x_delta_offset_cm) / 2.54
#     y_delta_offset_inch = float(y_delta_offset_cm) / 2.54

#     x_delta_offset_ratio = x_delta_offset_inch / fig_width_inch
#     y_delta_offset_ratio = y_delta_offset_inch / fig_width_inch

#     ## Determines updated bbox position
#     left = bbox.x0 + x_delta_offset_ratio
#     bottom = bbox.y0 + y_delta_offset_ratio
#     width = bbox.x1 - bbox.x0
#     height = bbox.y1 - bbox.y0

#     if dragh:
#         width -= x_delta_offset_ratio

#     if dragv:
#         height -= y_delta_offset_ratio

#     ax.set_pos(
#         [
#             left,
#             bottom,
#             width,
#             height,
#         ]
#     )

#     return ax
