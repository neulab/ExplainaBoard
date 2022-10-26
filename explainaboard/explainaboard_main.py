"""The main entry point for running ExplainaBoard."""

from __future__ import annotations

import argparse
import copy
import json
import os

import eaas.endpoint

from explainaboard import get_loader_class, get_processor_class, TaskType
from explainaboard.constants import Source
from explainaboard.info import SysOutputInfo
from explainaboard.loaders.file_loader import (
    DatalabLoaderOption,
    FileLoaderField,
    FileLoaderMetadata,
)
from explainaboard.metrics.eaas import EaaSMetricConfig
from explainaboard.metrics.metric import MetricConfig, Score
from explainaboard.serialization import common_registry
from explainaboard.utils.io_utils import text_writer
from explainaboard.utils.logging import get_logger
from explainaboard.utils.tensor_analysis import (
    aggregate_score_tensor,
    filter_score_tensor,
    print_score_tensor,
)
from explainaboard.utils.typing_utils import unwrap
from explainaboard.visualizers import get_pairwise_performance_gap
from explainaboard.visualizers.draw_charts import draw_charts_from_reports


def get_tasks(task: TaskType, system_outputs: list[str]) -> list[TaskType]:
    """Get the task for each system output.

    Args:
        task: Explicitly specified task. Use if present
        system_outputs: System output files, load from metadata in these files if
          an explicit task is not set

    Returns:
        A list of task types for each system
    """
    real_tasks: list[TaskType] = []
    if task:
        real_tasks = [task] * len(system_outputs)
        if task not in TaskType.list():
            raise ValueError(
                f"Task name {task} was not recognized. ExplainaBoard currently "
                f"supports:{TaskType.list()}"
            )
        return real_tasks
    else:
        for sys_output in system_outputs:
            # give me a task, or give me death (by exception)
            task_or_die: TaskType | None = None
            msg: str = ""
            try:
                metadata = FileLoaderMetadata.from_file(sys_output)
                task_or_die = TaskType(unwrap(metadata.task_name))
            except Exception as e:
                msg = str(e)
            if task_or_die is None:
                raise ValueError(
                    "Must either specify a task explicitly or have one "
                    "specified in metadata, but could find neither for "
                    f"{sys_output}. {msg}"
                )
            real_tasks.append(unwrap(task_or_die))
    return real_tasks


def analyze_reports(args):
    """Analyze reports based on the input arguments.

    score_tensor is a nested dict, for example
    score_tensor[system_name][dataset_name][language] =
    {
        'metric_name':
        'Accuracy',
        'value': 0.8802855573860516,
        'confidence_score_low': 0.8593406593406593,
        'confidence_score_high': 0.9
    }
    """
    reports = args.reports
    systems: list[str] | None = args.systems
    datasets: list[str] | None = args.datasets
    languages: list[str] | None = args.languages
    systems_aggregation: str | None = args.systems_aggregation
    datasets_aggregation: str | None = args.datasets_aggregation
    languages_aggregation: str | None = args.languages_aggregation
    score_tensor = {}
    for report in reports:
        with open(report) as fin:

            report_dict = json.load(fin)

            system_name = report_dict["system_name"]
            dataset_name = report_dict["dataset_name"]
            language = report_dict["language"]
            # TODO(Pengfei): So far, only one metric is considered
            metric = report_dict["metric_names"][0]
            score_info = report_dict["results"]["overall"][metric]

            if system_name not in score_tensor.keys():
                score_tensor[system_name] = {}
            if dataset_name not in score_tensor[system_name].keys():
                score_tensor[system_name][dataset_name] = {}
            if language not in score_tensor[system_name][dataset_name].keys():
                score_tensor[system_name][dataset_name][language] = {}
            score_tensor[system_name][dataset_name][language] = score_info

    # filter by three dimensions
    score_tensor_filter = filter_score_tensor(
        score_tensor, systems, datasets, languages
    )

    # aggregation by three dimensions
    score_tensor_aggregated = aggregate_score_tensor(
        score_tensor_filter,
        systems_aggregation,
        datasets_aggregation,
        languages_aggregation,
    )
    print_score_tensor(score_tensor_aggregated)


def create_parser():
    """Create the parser with argparse.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(description="Explainable Leaderboards for NLP")
    parser.add_argument("--task", type=str, required=False, help="the task name")
    parser.add_argument(
        "--system-outputs",
        type=str,
        required=True,
        nargs="+",
        help=(
            "the directories of system outputs. Multiple one should be separated by "
            "space, for example: system1 system2"
        ),
    )

    parser.add_argument(
        "--reports",
        type=str,
        required=False,
        nargs="+",
        help="the directories of analysis reports. Multiple one should be separated "
        "by space, for example: report1 report2",
    )

    parser.add_argument(
        "--systems",
        type=str,
        required=False,
        nargs="+",
        help="the list of system names",
    )

    parser.add_argument(
        "--datasets",
        type=str,
        required=False,
        nargs="+",
        help="the list of dataset names",
    )

    parser.add_argument(
        "--languages",
        type=str,
        required=False,
        nargs="+",
        help="the list of language names",
    )

    parser.add_argument(
        "--systems-aggregation",
        type=str,
        required=False,
        help="None|minus|combination",
    )

    parser.add_argument(
        "--datasets-aggregation",
        type=str,
        required=False,
        help="None|average|",
    )

    parser.add_argument(
        "--languages-aggregation",
        type=str,
        required=False,
        help="None|average|",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        default=None,
        help="the name of dataset",
    )

    parser.add_argument(
        "--sub-dataset",
        type=str,
        required=False,
        default=None,
        help="the name of sub-dataset",
    )

    parser.add_argument(
        "--split",
        type=str,
        required=False,
        default="test",
        help="the name of the split within the dataset",
    )

    parser.add_argument(
        "--language",
        "--target-language",
        dest="target_language",
        type=str,
        required=False,
        default=None,
        help="the language of system output, defaults to the default of the dataset if "
        "available, ",
    )

    parser.add_argument(
        "--source-language",
        type=str,
        required=False,
        default=None,
        help="language of system input",
    )

    parser.add_argument(
        "--no-use-cache",
        dest="use_cache",
        action="store_false",
        help="Disable cached statistics over training set.",
    )

    parser.add_argument(
        "--metrics",
        type=str,
        required=False,
        nargs="*",
        help="multiple metrics should be separated by space",
    )

    parser.add_argument(
        "--output-file-type",
        type=str,
        required=False,
        default=None,
        help="the file type: json, tsv, conll",
    )

    parser.add_argument(
        "--conf-value",
        dest="confidence_alpha",
        type=float,
        required=False,
        help="Deprecated. use --confidence-alpha instead.",
    )

    parser.add_argument(
        "--confidence-alpha",
        type=float,
        required=False,
        default=0.05,
        help=(
            "the *inverse* confidence level of confidence intervals. If you need to "
            "set the confidence level to 0.95, set this value to 0.05."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        default=None,
        help="the directory of output files",
    )

    parser.add_argument(
        "--report-json",
        type=str,
        required=False,
        default=None,
        help="the place to write the report json file",
    )

    parser.add_argument(
        "--system-details",
        type=str,
        required=False,
        help="a json file to store detailed information for a system",
    )

    parser.add_argument(
        "--custom-dataset-paths",
        type=str,
        nargs="*",
        help="path to custom dataset",
    )

    parser.add_argument(
        "--custom-dataset-file-type",
        type=str,
        help="file types for custom datasets",
    )

    parser.add_argument(
        "--skip-failed-analyses",
        action="store_true",
        help="whether to skip failed analyses or report errors.",
    )
    return parser


def get_metric_config_or_eaas(name: str) -> type[MetricConfig]:
    """Obtains MetricConfig class from registry or corresponding EaaS binding.

    Args:
        name: Name of the metric.

    Returns:
        A MetricConfig class associated to either a registered Metric or EaaS.

    Raises:
        ValueError: `name` is not registered in neither the registry nor EaaS.
    """
    try:
        cls = common_registry.get_type(name)
        if not issubclass(cls, MetricConfig):
            raise TypeError(f"Obtained class is not a MetricConfig: {cls.__name__}")
        return cls
    except ValueError:
        if name in eaas.endpoint.EndpointConfig().valid_metrics:
            return EaaSMetricConfig

    raise ValueError(
        f"Metric name {name} is not registered in neither the registry nor EaaS."
    )


# TODO(Pengfei): The overall implementation of this script should be deduplicated
def main():
    """The main function to be executed."""
    args = create_parser().parse_args()

    use_cache: bool = args.use_cache
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
                    raise ValueError(f"invalid json: {e} for system details")

        output_dir_figures = os.path.join(output_dir, "figures") if output_dir else None
        output_dir_reports = os.path.join(output_dir, "reports") if output_dir else None

        system_details: dict | None = load_system_details_path()
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if output_dir_figures and not os.path.exists(output_dir_figures):
            os.makedirs(output_dir_figures)
        if output_dir_reports and not os.path.exists(output_dir_reports):
            os.makedirs(output_dir_reports)

        # check for benchmark submission: explainaboard  --system-outputs ./data/
        # system_outputs/sst2/user_specified_metadata.json
        num_systems = len(system_outputs)
        dataset_file_types: list[str | None] = [dataset_file_type] * num_systems
        output_file_types: list[str | None] = [output_file_type] * num_systems
        custom_dataset_paths: list[str] | None = args.custom_dataset_paths
        dataset: str | None = args.dataset
        sub_dataset: str | None = args.sub_dataset
        split: str = args.split
        target_language: str = args.target_language
        source_language: str = args.source_language or target_language
        tasks = get_tasks(args.task, system_outputs)

        # Some loaders need to know the language of the inputs and outputs
        loader_field_mapping = {
            FileLoaderField.SOURCE_LANGUAGE: source_language,
            FileLoaderField.TARGET_LANGUAGE: target_language,
        }
        if custom_dataset_paths:  # load custom datasets
            loaders = [
                get_loader_class(task)(
                    dataset,
                    output,
                    Source.local_filesystem,
                    Source.local_filesystem,
                    dataset_file_type,
                    output_file_type,
                    field_mapping=loader_field_mapping,
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
                get_loader_class(task).from_datalab(
                    DatalabLoaderOption(dataset, sub_dataset, split=split),
                    sys_output,
                    Source.local_filesystem,
                    output_file_type,
                    field_mapping=loader_field_mapping,
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
                    f"Data must be identical for pairwise analysis, but length of "
                    "files "
                    f"{system_datasets[0]} ({num0}) != {system_datasets[1]} ({num1})"
                )

        # TODO(gneubig): This gets metadata from the first system and assumes it's the
        #  same for other systems
        target_language = (
            target_language or system_datasets[0].metadata.target_language or "en"
        )
        source_language = (
            source_language
            or system_datasets[0].metadata.source_language
            or target_language
        )

        # Setup metadata
        metadata = {
            "dataset_name": dataset,
            "sub_dataset_name": sub_dataset,
            "split_name": split,
            "source_language": source_language,
            "target_language": target_language,
            "confidence_alpha": args.confidence_alpha,
            "system_details": system_details,
            "custom_features": system_datasets[0].metadata.custom_features,
            "custom_analyses": system_datasets[0].metadata.custom_analyses,
        }
        if metric_names is not None:
            if "metric_configs" in metadata:
                raise ValueError("Cannot specify both metric names and metric configs")
            metric_configs = {
                name: get_metric_config_or_eaas(name)(
                    source_language=source_language,
                    target_language=target_language,
                )
                for name in metric_names
            }
            metadata["metric_configs"] = metric_configs

        # Run analysis
        reports: list[SysOutputInfo] = []
        for loader, system_dataset, system_full_path, task in zip(
            loaders, system_datasets, system_outputs, tasks
        ):
            metadata_copied = copy.deepcopy(metadata)
            metadata_copied["task_name"] = task

            processor = get_processor_class(task=task)()
            report = processor.process(
                metadata=metadata_copied,
                sys_output=system_dataset.samples,
                skip_failed_analyses=args.skip_failed_analyses,
                use_cache=use_cache,
            )
            reports.append(report)

            # print to the console
            logger = get_logger("report")

            logger.info("--- Overall Performance")
            for level_name, overall in report.results.overall.items():
                for metric_name, metric_result in overall.items():
                    value = metric_result.get_value(Score, "score").value
                    logger.info(f"{level_name}\t{metric_name}\t{value}")
            logger.info("")
            logger.info("--- Fine-grained Analyses")
            for analysis in report.results.analyses:
                if analysis is not None:
                    logger.info(analysis.generate_report())

            if output_dir:

                # save report to `output_dir_reports`
                x_file_name = os.path.basename(system_full_path).split(".")[0]
                report.write_to_directory(output_dir_reports, f"{x_file_name}.json")

                # generate figures and save them into  `output_dir_figures`
                if not os.path.exists(f"{output_dir_figures}/{x_file_name}"):
                    os.makedirs(f"{output_dir_figures}/{x_file_name}")
                draw_charts_from_reports(
                    [f"{output_dir_reports}/{x_file_name}.json"],
                    f"{output_dir_figures}/{x_file_name}",
                )

        with text_writer(args.report_json) as report_file:
            if len(system_outputs) == 1:  # individual system analysis
                reports[0].print_as_json(file=report_file)
            elif len(system_outputs) == 2:  # pairwise analysis
                compare_analysis = get_pairwise_performance_gap(reports[0], reports[1])
                compare_analysis.print_as_json(file=report_file)


if __name__ == "__main__":
    main()
