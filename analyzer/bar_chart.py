from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math




def df_format(data):
    data_df = []
    for entry in data:
        bucket_name = '{:.2f}'.format(entry["bucket_name"][0]) + ", " + '{:.1f}'.format(entry['bucket_name'][1]) if len(
            entry["bucket_name"]) == 2 else str(entry["bucket_name"][0])
        data_df.append([
            bucket_name,
            float(entry["value"]),
            0,
        ])

    # print(data_df)
    return data_df


def mp_format(data):
    data_mp = {
        "x": [],
        "y": [],
        "y_errormin": [],
        "y_errormax": [],
        "n_samples":[],
    }
    for entry in data:
        bucket_name = '{:.2f}'.format(entry["bucket_name"][0]) + ", " + '{:.2f}'.format(entry['bucket_name'][1]) if len(
            entry["bucket_name"]) == 2 else str(entry["bucket_name"][0])
        # data_df.append([
        #     bucket_name,
        #     float(entry["value"]),
        #     0,
        # ])

        data_mp["x"].append(bucket_name)
        data_mp["y"].append(float(entry["value"]))
        data_mp["y_errormin"].append(float(entry["confidence_score_low"]))
        data_mp["y_errormax"].append(float(entry["confidence_score_up"]))
        data_mp["n_samples"].append(entry["n_samples"])

    return data_mp


def get_ylim(y_list):
    low = min(y_list)
    high = max(y_list)
    delta = high - low
    return [low-0.1*delta, min(1,high+0.1*delta)]



def plot(data, save_path = None, x_label = "x", y_label = "y"):


    data_mp = mp_format(data)



    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.ylim(get_ylim(data_mp["y"]))
    # Reference for color design: https://www.colorhunt.co/
    clrs = ["#B4ECE3" if (x < max(data_mp["y"])) else '#FF8AAE' for x in data_mp["y"] ]
    plt.bar(data_mp["x"],
            data_mp["y"],
            width = 0.6,
            #color="#B4ECE3",
            color = clrs,
            edgecolor='#2D46B9',
            alpha=0.9,
            )

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.errorbar(data_mp["x"], data_mp["y"],
                 yerr=[data_mp["y_errormin"], data_mp["y_errormax"]],
                 fmt='o', color="r")  # you can use color ="r" for red or skip to default as blue

    for i in range(len(data_mp["x"])):
        plt.annotate(str(data_mp["n_samples"][i]),
                     xy=(data_mp["x"][i], data_mp["y"][i]),
                     xytext=(0,-12),
                     textcoords='offset points',
                     size = 15,
                     color='#2D46B9',
                     ha='center',
                     va='center')

    plt.xlabel(x_label, size = 15)
    plt.ylabel(y_label, size = 15)
    plt.xticks(size=10)
    plt.yticks(size=12)
    # plt.grid(True)

    plt.savefig(save_path)
    plt.close()









