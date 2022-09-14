"""draw_charts.py.

This is a program that takes in an ExplainaBoard report or reports and outputs visual
summaries of the included analyses.

Here is an example of usage:
> explainaboard --task text-classification --dataset sst2 \
                --system-outputs ./data/system_outputs/sst2/sst2-lstm-output.txt \
                --report-json report-lstm.json
> explainaboard --task text-classification --dataset sst2 \
                --system-outputs ./data/system_outputs/sst2/sst2-cnn-output.txt \
                --report-json report-cnn.json
> python -m explainaboard.visualizers.draw_charts \
                --reports report-lstm.json report-cnn.json

The output will be written to the `figures/` directory.
"""
from __future__ import annotations

import argparse
import json
import os

# NOTE(odashi): List is required to set the first argument of cast() in Python 3.8.
from typing import cast, List

from matplotlib import pyplot as plt
import numpy as np

from explainaboard.analysis.analyses import (
    AnalysisResult,
    BucketAnalysisResult,
    ComboCountAnalysisResult,
)
from explainaboard.analysis.performance import Performance
from explainaboard.info import SysOutputInfo
from explainaboard.utils.logging import progress
from explainaboard.utils.typing_utils import unwrap
from explainaboard.visualizers.bar_chart import make_bar_chart


def plot_combo_counts(
    combo_results: list[ComboCountAnalysisResult], output_dir: str, sys_names: list[str]
) -> None:
    """Plot combo count results.

    combo_results: The analyses to write out
    output_dir: The directory to write to
    sys_names: The names of the systems
    """
    # get feature maps
    if any(x.features != combo_results[0].features for x in combo_results):
        raise ValueError(
            f'all combo_results must have the same features, but got '
            f'{[x.features for x in combo_results]}'
        )
    feature_names = combo_results[0].features
    feature_maps: list[dict[str, int]] = [dict() for _ in feature_names]
    for combo_result in combo_results:
        for feats, count in unwrap(combo_result.combo_counts):
            for feat, featmap in zip(feats, feature_maps):
                featmap[feat] = featmap.get(feat, 0) + count
    # sort in descending order of frequency of each feature map
    sorted_names = [
        [v[0] for v in sorted(x.items(), key=lambda y: -y[1])] for x in feature_maps
    ]
    first_set = set(sorted_names[0])
    # if all sets of keys are the same, sort in descending order of total frequency
    if all(set(x) == first_set for x in sorted_names):
        total_cnt: dict[str, int] = {}
        for feature_map in feature_maps:
            for k, v in feature_map.items():
                total_cnt[k] = total_cnt.get(k, 0) + v
        total_sorted = [v[0] for v in sorted(total_cnt.items(), key=lambda y: -y[1])]
        sorted_names = [total_sorted for _ in sorted_names]
    feature_maps = [{v: i for i, v in enumerate(x)} for x in sorted_names]

    # Create all the plots
    for combo_idx, (combo_result, sys_name) in enumerate(zip(combo_results, sys_names)):
        if len(feature_names) != 2:
            raise ValueError(
                f'plot_combo_counts currently only supports feature combinations of '
                f'size 2, but got {feature_names}'
            )
        confusion_matrix = np.zeros([len(x) for x in feature_maps])
        for feats, count in unwrap(combo_result.combo_counts):
            confusion_matrix[
                feature_maps[0][feats[0]], feature_maps[1][feats[1]]
            ] = count
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(confusion_matrix, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                ax.text(
                    x=j,
                    y=i,
                    s=confusion_matrix[i, j],
                    va='center',
                    ha='center',
                    size='xx-large',
                )
        plt.ylabel(feature_names[0], fontsize=16)
        ax.set_yticks(np.arange(len(sorted_names[0])))
        ax.set_yticklabels(sorted_names[0])
        plt.xlabel(feature_names[1], fontsize=16)
        ax.set_xticks(np.arange(len(sorted_names[1])))
        ax.set_xticklabels(sorted_names[1])
        plt.title(f'confusion for {sys_name}', fontsize=18)
        out_file = os.path.join(
            output_dir, f'{feature_names[0]}_{feature_names[1]}_combo_{combo_idx}.png'
        )
        plt.savefig(out_file, format='png', bbox_inches='tight')


def render_interval_to_tick_label(interval: tuple[float, float]) -> str:
    """Render a bucket interval (tuple of floats) to a tick label to display.

    Args:
        interval: the input value range

    Returns:
        a string-rendered tick label
    """
    return f'[{interval[0]:.2f},{interval[0]:.2f}]'


def plot_buckets(
    bucket_results: list[BucketAnalysisResult], output_dir: str, sys_names: list[str]
) -> None:
    """Plot bucket results in a bar chart.

    Args:
        bucket_results: The analyses to write out
        output_dir: The directory to write to
        sys_names: The names of the systems
    """
    feature_name = bucket_results[0].name

    bucket0_ticklabels: list[str] = [
        x.bucket_name
        if x.bucket_name is not None
        else render_interval_to_tick_label(unwrap(x.bucket_interval))
        for x in bucket_results[0].bucket_performances
    ]
    bucket0_names = [
        x.metric_name for x in bucket_results[0].bucket_performances[0].performances
    ]
    for metric_id, metric_name in enumerate(bucket0_names):

        performances: list[list[Performance]] = [
            [x.performances[metric_id] for x in y.bucket_performances]
            for y in bucket_results
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
            xticklabels=bucket0_ticklabels,
            ylabel=metric_name,
        )


def draw_charts_from_reports(
    reports: list[str], output_dir: str, sys_names: list[str] | None = None
) -> None:
    """Draw bar charts from report file generated from ExplainaBoard.

    Args:
        reports: Reports to plot
        output_dir: The directory where the plots should be written
        sys_names: The names of the systems to write in the plots
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # TODO(gneubig): This should get the system name from inside the report
    if sys_names is None:
        sys_names = [os.path.splitext(os.path.basename(x))[0] for x in reports]
    elif len(sys_names) != len(reports):
        raise ValueError('Length of sys_names must equal that of reports')

    report_info: list[SysOutputInfo] = []
    for report in reports:
        with open(report) as fin:
            report_info.append(SysOutputInfo.from_dict(json.load(fin)))

    # --- Overall results
    num_levels = len(unwrap(report_info[0].analysis_levels))
    for level_id in range(num_levels):
        overall_results: list[list[Performance]] = [
            list(unwrap(x.results.overall)[level_id]) for x in report_info
        ]
        overall_metric_names = [x.metric_name for x in overall_results[0]]

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
            xticklabels=overall_metric_names,
            ylabel='metric value',
        )

    # --- analysis results
    analysis_results: list[list[AnalysisResult]] = [
        unwrap(x.results.analyses) for x in report_info
    ]
    if any(len(x) != len(analysis_results[0]) for x in analysis_results):
        raise ValueError(
            f'mismatched number of analyses: {[len(x) for x in analysis_results]}'
        )

    # Bucket performance: feature name, for example, sentence length
    for analysis_result in progress(zip(*analysis_results)):
        if any(x.name != analysis_result[0].name for x in analysis_result):
            raise ValueError(
                f'mismatched analyses: {[x.name for x in analysis_result]}'
            )

        analysis_result_list = list(analysis_result)

        if all(isinstance(x, BucketAnalysisResult) for x in analysis_result):
            plot_buckets(
                cast(List[BucketAnalysisResult], analysis_result_list),
                output_dir,
                sys_names,
            )
        elif all(isinstance(x, ComboCountAnalysisResult) for x in analysis_result):
            plot_combo_counts(
                cast(List[ComboCountAnalysisResult], analysis_result_list),
                output_dir,
                sys_names,
            )
        else:
            raise ValueError('illegal types of analyses')


def main():
    """Main function."""
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
        '--sys-names',
        type=str,
        required=False,
        nargs="+",
        default=None,
        help="names of each system, separated by spaces",
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=False,
        default="figures",
        help="the directory of generated figures",
    )

    args = parser.parse_args()

    reports = args.reports
    sys_names = args.sys_names
    output_dir = "figures" if args.output_dir is None else args.output_dir

    draw_charts_from_reports(reports, output_dir, sys_names=sys_names)


if __name__ == '__main__':
    main()
