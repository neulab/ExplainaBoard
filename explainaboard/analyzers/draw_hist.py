import argparse
import json
import os

from tqdm import tqdm

from explainaboard.analyzers.bar_chart import plot


def draw_bar_chart_from_report(report: str, output_dir: str) -> None:
    """
    Draw bar charts from report file generated from ExplainaBoard
    :param report:
    :param output_dir:
    :return:
    """

    with open(report) as fin:
        report_dict = json.load(fin)

    # read meta data from report
    metrics = report_dict["metric_names"]

    fine_grained_results = report_dict["results"]["fine_grained"]

    # print(fine_grained_results.keys())

    bar_charts = []
    for feature_name, buckets in tqdm(
        fine_grained_results.items()
    ):  # feature name, for example, sentence length
        for m in range(
            len(metrics)
        ):  # each bucket_info consists of multiple sub_buckets caculated by
            # different metrics (e.g, Accuracy, F1Score)
            bar_chart = []

            for (
                bucket_name,
                bucket_info,
            ) in (
                buckets.items()
            ):  # the number of buckets, for example, [1,5], [5,10], [10,15], [10,]
                """
                the structure of sub_bucket_info
                {
                    "metric_name":string
                    "value":string
                    "confidence_score_low":Optional[float]
                    "confidence_score_high":Optional[float]
                    "bucket_name":List[Any]
                    "n_samples":int
                }
                """

                bucket_info_revised = {
                    "n_samples": bucket_info["n_samples"],
                    "bucket_name": bucket_info["bucket_name"],
                    "metric_name": bucket_info["performances"][m]["metric_name"],
                    "value": bucket_info["performances"][m]["value"],
                    "confidence_score_low": bucket_info["performances"][m][
                        "confidence_score_low"
                    ],
                    "confidence_score_high": bucket_info["performances"][m][
                        "confidence_score_high"
                    ],
                }

                bar_chart.append(bucket_info_revised)
            bar_charts.append(bar_chart)

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            plot(
                bar_chart,
                save_path=output_dir + "/" + feature_name + "_" + metrics[m] + ".png",
                x_label=feature_name,
                y_label=metrics[m],
            )


def main():

    parser = argparse.ArgumentParser(
        description='Draw Histogram for ExplainaBoard Report'
    )

    parser.add_argument('--task', type=str, required=False, help="the task name")

    parser.add_argument(
        '--reports',
        type=str,
        required=True,
        nargs="+",
        help="the directories of reports. Multiple one should be separated by space, "
        "for example: report1 report2",
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        required=False,
        default="figures",
        help="the directory of generated figures",
    )

    args = parser.parse_args()

    reports = args.reports
    output_dir = "figures" if args.output_dir is None else args.output_dir

    draw_bar_chart_from_report(reports[0], output_dir)


if __name__ == '__main__':
    main()
    # explainaboard --system_outputs ./data/system_outputs/multilingual/json/
    # CL-mlpp15out1sum/marc/test-de_9330.json
    # python draw_hist.py --reports ../log.res
