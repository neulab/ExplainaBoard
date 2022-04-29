from __future__ import annotations

import argparse
import json
import os

from explainaboard.analyzers.bar_chart import make_bar_chart
from explainaboard.info import BucketPerformance, Performance, SysOutputInfo
from explainaboard.utils.logging import get_logger, progress
from explainaboard.utils.typing_utils import unwrap


def draw_bar_chart_from_reports(
    reports: list[str], output_dir: str, sys_names: list[str] | None = None
) -> None:
    """
    Draw bar charts from report file generated from ExplainaBoard
    :param reports: Reports to plot
    :param output_dir:
    :return:
    """

    # TODO(gneubig): This should get the system name from inside the report
    if sys_names is None:
        sys_names = [os.path.basename(x).replace('.json', '') for x in reports]
    elif len(sys_names) != len(reports):
        raise ValueError('Length of sys_names must equal that of reports')

    report_info: list[SysOutputInfo] = []
    for report in reports:
        with open(report) as fin:
            report_info.append(SysOutputInfo.from_dict(json.load(fin)))
    fg_results = [unwrap(x.results.fine_grained) for x in report_info]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # feature name, for example, sentence length
    for feature_name in progress(fg_results[0].keys()):
        # Make sure that buckets exist
        buckets: list[dict[str, BucketPerformance]] = []
        for i, fg_result in enumerate(fg_results):
            if feature_name not in fg_result:
                get_logger().error(f'error: feature {feature_name} not in {reports[i]}')
            else:
                buckets.append(fg_result[feature_name])
                bnames0, bnames = buckets[0].keys(), buckets[-1].keys()
                if len(bnames0) != len(bnames):
                    get_logger().error(
                        f'error: different number of buckets for {feature_name} in '
                        f'{reports[0]} and {reports[i]}'
                    )
                    buckets = []
                elif bnames0 != bnames:
                    get_logger().warning(
                        f'warning: different bucket labels for {feature_name} in '
                        f'{reports[0]} and {reports[i]}'
                    )
            if len(buckets) != i + 1:
                break
        if len(buckets) != len(reports):
            continue

        bucket0_names = list(buckets[0].keys())
        bucket0_values = list(buckets[0].values())
        bucket_metrics = [x.metric_name for x in bucket0_values[0].performances]
        for metric_id, metric_name in enumerate(bucket_metrics):

            performances: list[list[Performance]] = [
                [x.performances[metric_id] for x in y.values()] for y in buckets
            ]
            ys = [[x.value for x in y] for y in performances]

            y_errs = None
            if performances[0][0].confidence_score_low is not None:
                y_errs = [
                    (
                        [x.value - unwrap(x.confidence_score_low) for x in y],
                        [unwrap(x.confidence_score_high) - x.value for x in y],
                    )
                    for y in performances
                ]

            make_bar_chart(
                ys,
                output_dir,
                f'{feature_name}_{metric_name}',
                output_fig_format='png',
                fig_size=(8, 6),
                sys_names=sys_names,
                errs=y_errs,
                title=None,
                xlabel=feature_name,
                xticklabels=bucket0_names,
                ylabel=metric_name,
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
        '--sys_names',
        type=str,
        required=False,
        nargs="+",
        default=None,
        help="names of each system, separated by spaces",
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
    sys_names = args.sys_names
    output_dir = "figures" if args.output_dir is None else args.output_dir

    draw_bar_chart_from_reports(reports, output_dir, sys_names=sys_names)


if __name__ == '__main__':
    main()
    # explainaboard --system_outputs ./data/system_outputs/multilingual/json/
    # CL-mlpp15out1sum/marc/test-de_9330.json
    # python draw_hist.py --reports ../log.res
