from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math

# the original code is from: https://stackoverflow.com/questions/66895284/seaborn-barplot-with-specified-confidence-intervals


def df_format(data):
    data_df = []
    for entry in data:
        bucket_name = '{:.2f}'.format(entry["bucket_name"][0]) + ", " + '{:.2f}'.format(entry['bucket_name'][1]) if len(
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
    return [max(0,low-0.1*delta), min(1,high+0.1*delta)]



def plot(data, save_path = None):


    data_mp = mp_format(data)

    plt.ylim(get_ylim(data_mp["y"]))
    # print(get_ylim(data_mp["y"]))
    plt.bar(data_mp["x"], data_mp["y"], width = 0.5)


    plt.errorbar(data_mp["x"], data_mp["y"],
                 yerr=[data_mp["y_errormin"], data_mp["y_errormax"]],
                 fmt='o', color="r")  # you can use color ="r" for red or skip to default as blue

    for i in range(len(data_mp["x"])):
        plt.annotate(str(data_mp["n_samples"][i]),
                     xy=(data_mp["x"][i], data_mp["y"][i]),
                     ha='center',
                     va='bottom')

    plt.savefig(save_path)
    plt.close()









