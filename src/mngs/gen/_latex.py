#!/usr/bin/env python3


def to_latex_style(str_or_num):
    """
    Example:
        print(to_latex__style('aaa'))
        # '$aaa$'
    """
    string = str(str_or_num)
    if (string[0] == "$") and (string[-1] == "$"):
        return string
    else:
        return "${}$".format(string)


def add_hat_in_latex_style(str_or_num):
    """
    Example:
        print(add_hat_in_latex__style('aaa'))
        # '$\\hat{aaa}$'
    """
    return to_latex_style(r"\hat{%s}" % str_or_num)
