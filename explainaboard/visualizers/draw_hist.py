from __future__ import annotations

import argparse
import json
import os

from explainaboard.analysis.analyses import BucketAnalysisResult
from explainaboard.analysis.performance import Performance
from explainaboard.info import SysOutputInfo
from explainaboard.utils.logging import progress
from explainaboard.utils.typing_utils import unwrap
from explainaboard.visualizers.bar_chart import make_bar_chart


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

    num_levels = len(unwrap(report_info[0].analysis_levels))
    for level_id in range(num_levels):

        level_name = unwrap(report_info[0].analysis_levels)[level_id].name

        overall_results: list[list[Performance]] = [
            list(unwrap(x.results.overall)[level_id]) for x in report_info
        ]
        bucket_results: list[list[BucketAnalysisResult]] = [
            [
                y
                for y in unwrap(x.results.analyses)
                if isinstance(y, BucketAnalysisResult) and y.level == level_name
            ]
            for x in report_info
        ]
        bucket_names: list[list[str]] = [[y.name for y in x] for x in bucket_results]
        metric_names: list[list[str]] = [
            [y.metric_name for y in x] for x in overall_results
        ]
        for name, over, buck, buck_name, my_metrics in zip(
            sys_names, overall_results, bucket_results, bucket_names, metric_names
        ):
            if len(over) != len(overall_results[0]):
                raise ValueError(
                    f'mismatched overall results in {name} and {sys_names[0]}'
                )
            if len(buck) != len(bucket_results[0]):
                raise ValueError(f'mismatched buckets in {name} and {sys_names[0]}')
            if buck_name != bucket_names[0]:
                raise ValueError(
                    f'mismatched bucket names in {name} and '
                    f'{sys_names[0]}:\n{buck_name}\n{bucket_names[0]}'
                )
            if my_metrics != metric_names[0]:
                raise ValueError(
                    f'mismatched metric names in {name} and '
                    f'{sys_names[0]}:\n{my_metrics}\n{metric_names[0]}'
                )

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Overall performance
        ys = [[x.value for x in y] for y in overall_results]
        y_errs = None
        if overall_results[0][0].confidence_score_low is not None:
            y_errs = [
                (
                    [x.value - unwrap(x.confidence_score_low) for x in y],
                    [unwrap(x.confidence_score_high) - x.value for x in y],
                )
                for y in overall_results
            ]

        make_bar_chart(
            ys,
            output_dir,
            'overall',
            output_fig_format='png',
            fig_size=(8, 6),
            sys_names=sys_names,
            errs=y_errs,
            title=None,
            xticklabels=metric_names[0],
            ylabel='metric value',
        )

        # Bucket performance: feature name, for example, sentence length
        for analysis_idx, fg_results in progress(enumerate(zip(*bucket_results))):
            feature_name = bucket_names[0][analysis_idx]
            ba_results: list[BucketAnalysisResult] = fg_results

            bucket0_intervals = [
                x.bucket_interval for x in ba_results[0].bucket_performances
            ]
            for metric_id, metric_name in enumerate(metric_names[0]):

                performances: list[list[Performance]] = [
                    [x.performances[metric_id] for x in y.bucket_performances]
                    for y in ba_results
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
                    xticklabels=bucket0_intervals,
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
