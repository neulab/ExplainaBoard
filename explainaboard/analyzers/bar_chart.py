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


# def plot_chart_from_buckets(
#     buckets: list[dict[str, BucketPerformance]],
#     metric_id: int,
#     save_path: Optional[str] = None,
#     x_label: str = "x",
#     y_label: str = "y",
#     legend_names: Optional[list[str]] = None,
# ) -> None:
#     """
#     Generate bar chart based on given data and x,y labels
#     :param buckets: a list of buckets. note that these are serialized
#                     BucketPerformance objects
#     :param metric_id: which metric to draw
#     :param save_path:
#     :param x_label:
#     :param y_label:
#     :return:
#     """
#
#     performances: list[list[Performance]] = [[x.performances[metric_id] for x in
#                        y.values()] for y in buckets]
#     bucket_names = list(buckets[0].keys())
#     ys = [[x.value for x in y] for y in performances]
#
#     fig, ax = plt.subplots()
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#
#     if performances[0][0].confidence_score_low is not None:
#         y_low = [[x.confidence_score_low for x in y] for y in performances]
#         y_high = [[x.confidence_score_high for x in y] for y in performances]
#         plt.ylim((min(y_low), max(y_high)))
#
#     # Reference for color design: https://www.colorhunt.co/
#     max_y = max(ys)
#     clrs = ["#B4ECE3" if (x < max_y) else '#FF8AAE' for x in ys]
#     plt.bar(
#         bucket_names,
#         ys,
#         width=0.6,
#         # color="#B4ECE3",
#         color=clrs,
#         edgecolor='#2D46B9',
#         alpha=0.9,
#     )
#
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#
#     # for bucket_name, bucket_info in buckets.items():
#     #     plt.annotate(
#     #         str(bucket_info['n_samples']),
#     #         xy=(bucket_name, bucket_info['performances'][metric_id]['value']),
#     #         xytext=(0, -12),
#     #         textcoords='offset points',
#     #         size=15,
#     #         color='#2D46B9',
#     #         ha='center',
#     #         va='center',
#     #     )
#
#     plt.xlabel(x_label, size=15)
#     plt.ylabel(y_label, size=15)
#     plt.xticks(size=10)
#     plt.yticks(size=12)
#     # plt.grid(True)
#
#     fig.tight_layout()  # control left and right margin of the figure
#     plt.savefig(save_path)
#     plt.close()
