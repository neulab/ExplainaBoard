"""Utility functions to create bar charts."""

from __future__ import annotations

import logging
import os

from matplotlib import pyplot as plt
import numpy as np

logging.basicConfig(level="INFO")

mlogger = logging.getLogger("matplotlib")
mlogger.setLevel(logging.WARNING)


def get_ylim(y_list: list) -> tuple[float, float]:
    """Get the upper/lower bound for y axis.

    Args:
        y_list: A list of y axis values.

    Returns:
        A two-element tuple of the bottom and top of the range
    """
    low = min(y_list)
    high = max(y_list)
    delta = high - low
    return (low - 0.1 * delta, min(1, high + 0.1 * delta))


bar_colors = [
    "#7293CB",
    "#E1974C",
    "#84BA5B",
    "#D35E60",
    "#808585",
    "#9067A7",
    "#AB6857",
    "#CCC210",
]


def make_bar_chart(
    datas: list[list[float]],
    output_directory: str,
    output_fig_file: str,
    output_fig_format: str = "png",
    fig_size: tuple[int, int] = (8, 6),
    sys_names: list[str] | None = None,
    errs: list[tuple[list[float], list[float]]] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    xticklabels: list[str] | None = None,
    ylabel: str | None = None,
) -> None:
    """Create a bar chart given the inputs.

    Args:
        datas: A list of lists of floats containing the data of the bar chart
        output_directory: The directory to write the outputs to
        output_fig_file: The name of the figure file
        output_fig_format: The format/file extension (e.g. png or pdf)
        fig_size: The size of the figure
        sys_names: The names of the systems to put in the legend
        errs: Bounds for error bars above and below the actual data value
        title: The title of the figure
        xlabel: The label of the x axis
        xticklabels: The label of each group of bars on the x axis
        ylabel: The label of the y axis
    """
    fig, ax = plt.subplots(figsize=fig_size)
    ind = np.arange(len(datas[0]))
    width = 0.7 / len(datas)
    bars = []
    for i, data in enumerate(datas):
        err = errs[i] if errs is not None else None
        bars.append(
            ax.bar(
                ind + i * width, data, width, color=bar_colors[i], bottom=0, yerr=err
            )
        )
    # Set axis/title labels
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xticklabels is not None:
        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(xticklabels)
        plt.xticks(rotation=70)
    else:
        ax.xaxis.set_visible(False)

    if sys_names is not None:
        ax.legend(bars, sys_names)
    ax.autoscale_view()

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    out_file = os.path.join(output_directory, f"{output_fig_file}.{output_fig_format}")
    plt.savefig(out_file, format=output_fig_format, bbox_inches="tight")
