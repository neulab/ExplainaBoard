from __future__ import annotations

import logging
from typing import Optional

from matplotlib import pyplot as plt

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


def plot_chart_from_buckets(
    buckets: dict[str, dict],
    metric_id: int,
    save_path: Optional[str] = None,
    x_label: str = "x",
    y_label: str = "y",
) -> None:
    """
    Generate bar chart based on given data and x,y labels
    :param buckets: a list of buckets. note that these are serialized BucketPerformance
                    objects
    :param metric_id: which metric to draw
    :param save_path:
    :param x_label:
    :param y_label:
    :return:
    """

    performances: list[dict] = [x['performances'][metric_id] for x in buckets.values()]
    bucket_names = list(buckets.keys())
    ys = [x['value'] for x in performances]

    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if performances[0]['confidence_score_low'] is not None:
        y_low = [x['confidence_score_low'] for x in performances]
        y_high = [x['confidence_score_high'] for x in performances]
        plt.ylim((min(y_low), max(y_high)))

    # Reference for color design: https://www.colorhunt.co/
    max_y = max(ys)
    clrs = ["#B4ECE3" if (x < max_y) else '#FF8AAE' for x in ys]
    plt.bar(
        bucket_names,
        ys,
        width=0.6,
        # color="#B4ECE3",
        color=clrs,
        edgecolor='#2D46B9',
        alpha=0.9,
    )

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    for bucket_name, bucket_info in buckets.items():
        plt.annotate(
            str(bucket_info['n_samples']),
            xy=(bucket_name, bucket_info['performances'][metric_id]['value']),
            xytext=(0, -12),
            textcoords='offset points',
            size=15,
            color='#2D46B9',
            ha='center',
            va='center',
        )

    plt.xlabel(x_label, size=15)
    plt.ylabel(y_label, size=15)
    plt.xticks(size=10)
    plt.yticks(size=12)
    # plt.grid(True)

    fig.tight_layout()  # control left and right margin of the figure
    plt.savefig(save_path)
    plt.close()
