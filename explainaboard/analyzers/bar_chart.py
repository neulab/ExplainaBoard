from __future__ import annotations

import logging
from typing import Any, Optional

from matplotlib import pyplot as plt

logging.basicConfig(level='INFO')

mlogger = logging.getLogger('matplotlib')
mlogger.setLevel(logging.WARNING)


def mp_format(data: list[dict[str, Any]]) -> dict:
    """
    Adapt the format of data
    :param data:
    :return:
    """
    data_mp: dict = {
        "x": [],
        "y": [],
        "y_errormin": [],
        "y_errormax": [],
        "n_samples": [],
    }
    for entry in data:

        bucket_name = (
            '{:.2f}'.format(entry["bucket_name"][0])
            + ", "
            + '{:.2f}'.format(entry['bucket_name'][1])
            if len(entry["bucket_name"]) == 2
            else str(entry["bucket_name"][0])
        )
        # data_df.append([
        #     bucket_name,
        #     float(entry["value"]),
        #     0,
        # ])

        confidence_score_low = (
            0
            if entry["confidence_score_low"] is None
            else float(entry["confidence_score_low"])
        )
        confidence_score_high = (
            0
            if entry["confidence_score_high"] is None
            else float(entry["confidence_score_high"])
        )
        value = float('{:.2f}'.format(entry["value"]))

        data_mp["x"].append(bucket_name)
        data_mp["y"].append(value)

        # print(confidence_score_low, confidence_score_high)

        data_mp["y_errormin"].append(value - confidence_score_low)
        data_mp["y_errormax"].append(confidence_score_high - value)
        data_mp["n_samples"].append(entry["n_samples"])

    return data_mp


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


def plot(
    data: list[dict[str, Any]],
    save_path: Optional[str] = None,
    x_label: str = "x",
    y_label: str = "y",
) -> None:
    """
    Generate bar chart based on given data and x,y labels
    :param data:
    :param save_path:
    :param x_label:
    :param y_label:
    :return:
    """

    data_mp = mp_format(data)

    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.ylim(get_ylim(data_mp["y"] + data_mp["y_errormin"] + data_mp["y_errormax"]))
    # Reference for color design: https://www.colorhunt.co/
    clrs = ["#B4ECE3" if (x < max(data_mp["y"])) else '#FF8AAE' for x in data_mp["y"]]
    plt.bar(
        data_mp["x"],
        data_mp["y"],
        width=0.6,
        # color="#B4ECE3",
        color=clrs,
        edgecolor='#2D46B9',
        alpha=0.9,
    )

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    for i in range(len(data_mp["x"])):
        plt.annotate(
            str(data_mp["n_samples"][i]),
            xy=(data_mp["x"][i], data_mp["y"][i]),
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
