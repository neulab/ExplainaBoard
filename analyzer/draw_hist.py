import json
from bar_chart import plot
from pathlib import Path
import os
from tqdm import tqdm


def draw_bar_chart_from_report(path_report, path_fig):



    with open(path_report) as fin:
        report_dict = json.load(fin)


    # read meta data from report
    task_name = report_dict["task_name"]
    dataset_name = report_dict["dataset_name"]
    metrics = report_dict["metric_names"]

    fine_grained_results = report_dict["results"]["fine_grained"]

    # print(fine_grained_results.keys())


    bar_charts = []
    for feature_name, buckets in tqdm(fine_grained_results.items()): # feature name, for example, sentence length
        for m in range(len(metrics)): # each bucket_info consists of multiple sub_buckets caculated by different metrics (e.g, Accuracy, F1Score)
            bar_chart = []
            for bucket_info in buckets: # the number of buckets, for example, [1,5], [5,10], [10,15], [10,]
                """
                the structure of sub_bucket_info
                {
                    "metric_name":string
                    "value":string
                    "confidence_score_low":string
                    "confidence_score_up":string
                    "bucket_name":List[Any]
                    "n_samples":int
                    "bucket_samples":List[string]
                }
                """
                bar_chart.append(bucket_info[m])
            bar_charts.append(bar_chart)

            if not os.path.exists(path_fig):
                os.makedirs(path_fig)
            plot(bar_chart, save_path=path_fig + "/" + feature_name + "_" + metrics[m] + ".png", x_label = feature_name, y_label = metrics[m])






file_path = "./report-tc.json"
file_path = "./report-cmp.json"
draw_bar_chart_from_report(file_path, "fig")

