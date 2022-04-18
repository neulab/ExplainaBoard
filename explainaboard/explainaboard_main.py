from __future__ import annotations

import argparse
import json
import os
import sys

from explainaboard import (
    FileType,
    get_custom_dataset_loader,
    get_datalab_loader,
    get_pairwise_performance_gap,
    get_processor,
    TaskType,
)
from explainaboard.analyzers.draw_hist import draw_bar_chart_from_report
from explainaboard.constants import Source
from explainaboard.info import SysOutputInfo
from explainaboard.loaders.file_loader import DatalabLoaderOption
from explainaboard.utils.tensor_analysis import (
    aggregate_score_tensor,
    filter_score_tensor,
    print_score_tensor,
)


def get_tasks(task: str, system_outputs: list[str]) -> list[str]:
    """TODO: TaskType.text_classification is set for temporal use, this needs to be
    fixed
    """
    real_tasks: list[str]
    if task:
        real_tasks = [task] * len(system_outputs)
        if task not in TaskType.list():
            raise ValueError(
                f'Task name {task} was not recognized. ExplainaBoard currently '
                f'supports:{TaskType.list()}'
            )
        return real_tasks
    else:
        file_type = FileType.json
        dummy_task = TaskType.text_classification
        for sys_output in system_outputs:
            loader = get_custom_dataset_loader(
                dummy_task,
                sys_output,
                sys_output,
                dataset_file_type=file_type,
                output_file_type=file_type,
            )
            if not loader.user_defined_metadata_configs:
                raise ValueError(
                    f"user_defined_metadata_configs in system output {sys_output} "
                    "hasn't been specified or task name should be specified"
                )
            real_tasks.append(loader.user_defined_metadata_configs['task_name'])
    return real_tasks


def analyze_reports(args):
    """
    score_tensor is a nested dict, for example
    score_tensor[model_name][dataset_name][language] =
    {
        'metric_name':
        'Accuracy',
        'value': 0.8802855573860516,
        'confidence_score_low': 0.8593406593406593,
        'confidence_score_high': 0.9
    }
    """
    reports = args.reports
    models: list[str] | None = args.models
    datasets: list[str] | None = args.datasets
    languages: list[str] | None = args.languages
    models_aggregation: str | None = args.models_aggregation
    datasets_aggregation: str | None = args.datasets_aggregation
    languages_aggregation: str | None = args.languages_aggregation
    score_tensor = {}
    for report in reports:
        with open(report) as fin:

            report_dict = json.load(fin)

            model_name = report_dict["model_name"]
            dataset_name = report_dict["dataset_name"]
            language = report_dict["language"]
            # TODO(Pengfei): So far, only one metric is considered
            metric = report_dict["metric_names"][0]
            score_info = report_dict["results"]["overall"][metric]

            # print(model_name, dataset_name, language)

            if model_name not in score_tensor.keys():
                score_tensor[model_name] = {}
            if dataset_name not in score_tensor[model_name].keys():
                score_tensor[model_name][dataset_name] = {}
            if language not in score_tensor[model_name][dataset_name].keys():
                score_tensor[model_name][dataset_name][language] = {}
            score_tensor[model_name][dataset_name][language] = score_info

    # filter by three dimensions
    score_tensor_filter = filter_score_tensor(score_tensor, models, datasets, languages)

    # aggregation by three dimensions
    score_tensor_aggregated = aggregate_score_tensor(
        score_tensor_filter,
        models_aggregation,
        datasets_aggregation,
        languages_aggregation,
    )
    print_score_tensor(score_tensor_aggregated)


def create_parser():
    parser = argparse.ArgumentParser(description='Explainable Leaderboards for NLP')
    parser.add_argument('--task', type=str, required=False, help="the task name")
    parser.add_argument(
        '--system_outputs',
        type=str,
        required=True,
        nargs="+",
        help=(
            "the directories of system outputs. Multiple one should be separated by "
            "space, for example: system1 system2"
        ),
    )

    parser.add_argument(
        '--reports',
        type=str,
        required=False,
        nargs="+",
        help="the directories of analysis reports. Multiple one should be separated "
        "by space, for example: report1 report2",
    )

    parser.add_argument(
        '--models',
        type=str,
        required=False,
        nargs="+",
        help="the list of model names",
    )

    parser.add_argument(
        '--datasets',
        type=str,
        required=False,
        nargs="+",
        help="the list of dataset names",
    )

    parser.add_argument(
        '--languages',
        type=str,
        required=False,
        nargs="+",
        help="the list of language names",
    )

    parser.add_argument(
        '--models_aggregation',
        type=str,
        required=False,
        help="None|minus|combination",
    )

    parser.add_argument(
        '--datasets_aggregation',
        type=str,
        required=False,
        help="None|average|",
    )

    parser.add_argument(
        '--languages_aggregation',
        type=str,
        required=False,
        help="None|average|",
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=False,
        default=None,
        help="the name of dataset",
    )

    parser.add_argument(
        '--sub_dataset',
        type=str,
        required=False,
        default=None,
        help="the name of sub-dataset",
    )

    parser.add_argument(
        '--language',
        type=str,
        required=False,
        default="en",
        help="the language of system output",
    )

    parser.add_argument(
        '--reload_stat',
        type=str,
        required=False,
        default=None,
        help="reload precomputed statistics over training set (if exists)",
    )

    parser.add_argument(
        '--metrics',
        type=str,
        required=False,
        nargs="*",
        help="multiple metrics should be separated by space",
    )

    parser.add_argument(
        '--output_file_type',
        type=str,
        required=False,
        default=None,
        help="the file type: json, tsv, conll",
    )

    parser.add_argument(
        '--conf_value',
        type=float,
        required=False,
        default=0.05,
        help="the p-value with which to calculate the confidence interval",
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        required=False,
        default="output",
        help="the directory of output files",
    )

    parser.add_argument(
        '--report_json',
        type=str,
        required=False,
        default=None,
        help="the place to write the report json file",
    )

    parser.add_argument(
        '--system_details',
        type=str,
        required=False,
        help="a json file to store detailed information for a system",
    )

    parser.add_argument(
        '--custom_dataset_paths',
        type=str,
        nargs="*",
        help="path to custom dataset",
    )

    parser.add_argument(
        '--custom_dataset_file_type',
        type=str,
        help="file types for custom datasets",
    )
    return parser


# TODO(Pengfei): The overall implementation of this script should be deduplicated
def main():
    args = create_parser().parse_args()

    reload_stat: bool = False if args.reload_stat == "0" else True
    system_outputs: list[str] = args.system_outputs

    reports: list[str] | None = args.reports
    metric_names: list[str] | None = args.metrics
    dataset_file_type: str | None = args.custom_dataset_file_type
    output_file_type: str | None = args.output_file_type
    output_dir: str = args.output_dir

    # If reports have been specified, ExplainaBoard cli will perform analysis
    # over report files.
    if args.reports:
        analyze_reports(args)
    else:

        def load_system_details_path():
            if args.system_details:
                try:
                    with open(args.system_details) as fin:
                        return json.load(fin)
                except ValueError as e:
                    print('invalid json: %s for system details' % e)

        output_dir_figures = os.path.join(output_dir, "figures")
        output_dir_reports = os.path.join(output_dir, "reports")

        def setup_output_folders():
            """Setup for generated reports and figures"""
            # This part could be generalized
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if not os.path.exists(output_dir_figures):
                os.makedirs(output_dir_figures)
            if not os.path.exists(output_dir_reports):
                os.makedirs(output_dir_reports)

        system_details: dict | None = load_system_details_path()
        setup_output_folders()

        # check for benchmark submission: explainaboard  --system_outputs ./data/
        # system_outputs/sst2/user_specified_metadata.json
        num_systems = len(system_outputs)
        tasks = get_tasks(args.task, system_outputs)
        dataset_file_types: list[str | None] = [dataset_file_type] * num_systems
        output_file_types: list[str | None] = [output_file_type] * num_systems
        custom_dataset_paths: list[str] | None = args.custom_dataset_paths
        dataset: str | None = args.dataset
        sub_dataset: str | None = args.sub_dataset

        if custom_dataset_paths:  # load custom datasets
            loaders = [
                get_custom_dataset_loader(
                    task,
                    dataset,
                    output,
                    Source.local_filesystem,
                    Source.local_filesystem,
                    dataset_file_type,
                    output_file_type,
                )
                for task, dataset, output, dataset_file_type, output_file_type in zip(
                    tasks,
                    custom_dataset_paths,
                    system_outputs,
                    dataset_file_types,
                    output_file_types,
                )
            ]
        else:  # load from datalab
            if not dataset:
                raise ValueError("neither custom_dataset_paths or dataset is defined")
            loaders = [
                get_datalab_loader(
                    task,
                    DatalabLoaderOption(dataset, sub_dataset),
                    sys_output,
                    Source.local_filesystem,
                    output_file_type,
                )
                for task, sys_output, output_file_type in zip(
                    tasks, system_outputs, output_file_types
                )
            ]
        system_datasets = [loader.load() for loader in loaders]

        # validation
        if len(system_datasets) == 2:
            if len(system_datasets[0]) != len(system_datasets[1]):
                num0 = len(system_datasets[0])
                num1 = len(system_datasets[1])
                raise ValueError(
                    f'Data must be identical for pairwise analysis, but length of '
                    'files '
                    f'{system_datasets[0]} ({num0}) != {system_datasets[1]} ({num1})'
                )
            if (
                loaders[0].user_defined_features_configs
                != loaders[1].user_defined_features_configs
            ):
                raise ValueError(
                    "User defined features must be the same for pairwise analysis."
                )

        # Setup metadata
        metadata = {
            "dataset_name": dataset,
            "sub_dataset_name": sub_dataset,
            "language": args.language,
            "reload_stat": reload_stat,
            "conf_value": args.conf_value,
            "system_details": system_details,
        }

        if metric_names is not None:
            metadata["metric_names"] = metric_names

        # Run analysis
        reports: list[SysOutputInfo] = []
        for loader, system_dataset, system_full_path, task in zip(
            loaders, system_datasets, system_outputs, tasks
        ):

            metadata.update(loader.user_defined_metadata_configs)
            metadata[
                "user_defined_features_configs"
            ] = loader.user_defined_features_configs
            metadata["task_name"] = task

            report = get_processor(task=task).process(
                metadata=metadata, sys_output=system_dataset
            )
            reports.append(report)

            # save report to `output_dir_reports`
            x_file_name = os.path.basename(system_full_path).split(".")[0]
            report.write_to_directory(output_dir_reports, f"{x_file_name}.json")

            # generate figures and save them into  `output_dir_figures`
            if not os.path.exists(f"{output_dir_figures}/{x_file_name}"):
                os.makedirs(f"{output_dir_figures}/{x_file_name}")
            draw_bar_chart_from_report(
                f"{output_dir_reports}/{x_file_name}.json",
                f"{output_dir_figures}/{x_file_name}",
            )

        if args.report_json is not None:
            report_file = open(args.report_json, 'w')
        else:
            report_file = sys.stdout
        if len(system_outputs) == 1:  # individual system analysis
            reports[0].print_as_json(file=report_file)
        elif len(system_outputs) == 2:  # pairwise analysis
            compare_analysis = get_pairwise_performance_gap(
                reports[0].to_dict(), reports[1].to_dict()
            )
            json.dump(compare_analysis, fp=report_file, indent=2)
        if args.report_json is not None:
            report_file.close()


if __name__ == '__main__':
    main()
