from __future__ import annotations

import logging
import os

from matplotlib import pyplot as plt
import numpy as np

logging.basicConfig(level='INFO')

mlogger = logging.getLogger('matplotlib')
mlogger.setLevel(logging.WARNING)


def get_ylim(y_list: list) -> list:
    """
    get the upper/lower bound for y axis
    :param y_list:
    :return:
    """
    low = min(y_list)
    high = max(y_list)
    delta = high - low
    return [low - 0.1 * delta, min(1, high + 0.1 * delta)]


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
    datas,
    output_directory,
    output_fig_file,
    output_fig_format='png',
    fig_size=300,
    sys_names=None,
    errs=None,
    title=None,
    xlabel=None,
    xticklabels=None,
    ylabel=None,
):
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
    out_file = os.path.join(output_directory, f'{output_fig_file}.{output_fig_format}')
    plt.savefig(out_file, format=output_fig_format, bbox_inches='tight')
