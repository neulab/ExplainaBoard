import argparse
import json
import os

from tqdm import tqdm

from explainaboard.analyzers.bar_chart import plot_chart_from_buckets


def draw_bar_chart_from_report(report: str, output_dir: str) -> None:
    """
    Draw bar charts from report file generated from ExplainaBoard
    :param report:
    :param output_dir:
    :return:
    """

    with open(report) as fin:
        report_dict = json.load(fin)

    fine_grained_results = report_dict["results"]["fine_grained"]

    # print(fine_grained_results.keys())
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for feature_name, buckets in tqdm(
        fine_grained_results.items()
    ):  # feature name, for example, sentence length
        bucket_values = list(buckets.values())
        bucket_metrics = [x['metric_name'] for x in bucket_values[0]['performances']]
        for metric_id, metric_name in enumerate(bucket_metrics):

            plot_chart_from_buckets(
                buckets,
                metric_id,
                save_path=f'{output_dir}/{feature_name}_{metric_name}.png',
                x_label=feature_name,
                y_label=metric_name,
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
