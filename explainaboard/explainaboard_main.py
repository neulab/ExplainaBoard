import argparse
import json
import os

from explainaboard import (
    FileType,
    get_loader,
    get_pairwise_performance_gap,
    get_processor,
    TaskType,
)
from explainaboard.analyzers.draw_hist import draw_bar_chart_from_report
from explainaboard.utils.tensor_analysis import (
    aggregate_score_tensor,
    filter_score_tensor,
    print_score_tensor,
)


# TODO(Pengfei): The overall implementation of this script should be deduplicated
def main():

    parser = argparse.ArgumentParser(description='Explainable Leaderboards for NLP')

    parser.add_argument('--task', type=str, required=False, help="the task name")

    parser.add_argument(
        '--system_outputs',
        type=str,
        required=False,
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
        '--type',
        type=str,
        required=False,
        default="single",
        help="analysis type: single|pair|combine",
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
        '--file_type',
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
        '--system_details',
        type=str,
        required=False,
        help="a json file to store detailed information for a system",
    )

    args = parser.parse_args()

    dataset = args.dataset
    sub_dataset = args.sub_dataset
    language = args.language
    task = args.task
    reload_stat = False if args.reload_stat == "0" else True
    system_outputs = args.system_outputs

    reports = args.reports
    metric_names = args.metrics
    file_type = args.file_type
    output_dir = args.output_dir
    models = args.models
    datasets = args.datasets
    languages = args.languages
    models_aggregation = args.models_aggregation
    datasets_aggregation = args.datasets_aggregation
    languages_aggregation = args.languages_aggregation

    system_details_path = args.system_details

    # get system_details from input json file
    system_details = None
    if system_details_path is not None:
        try:
            with open(system_details_path) as fin:
                system_details = json.load(fin)
        except ValueError as e:
            print('invalid json: %s' % e)

    # If reports have been specified, ExplainaBoard cli will perform analysis
    # over report files.
    if reports is not None:

        """
        score_tensor is a nested dict, for exampple
        score_tensor[model_name][dataset_name][language] =
        {
            'metric_name':
            'Accuracy',
            'value': 0.8802855573860516,
            'confidence_score_low': 0.8593406593406593,
            'confidence_score_high': 0.9
        }
        """
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
        score_tensor_filter = filter_score_tensor(
            score_tensor, models, datasets, languages
        )

        # aggregation by three dimensions
        score_tensor_aggregated = aggregate_score_tensor(
            score_tensor_filter,
            models_aggregation,
            datasets_aggregation,
            languages_aggregation,
        )
        print_score_tensor(score_tensor_aggregated)

        return True

    # Setup for generated reports and figures
    output_dir_figures = output_dir + "/" + "figures"
    output_dir_reports = output_dir + "/" + "reports"

    # This part could be generalized
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_dir_figures):
        os.makedirs(output_dir_figures)
    if not os.path.exists(output_dir_reports):
        os.makedirs(output_dir_reports)

    # check for benchmark submission: explainaboard  --system_outputs ./data/
    # system_outputs/sst2/user_specified_metadata.json
    real_tasks = [] if task is None else [task]
    if task is None:
        file_type = "json"
        dummy_task = TaskType.text_classification
        # TaskType.text_classification is set for temporal use, and this need to be
        # generalized
        for x in system_outputs:
            loader = get_loader(dummy_task, data=x, file_type=file_type)
            if (
                loader.user_defined_metadata_configs is None
                or len(loader.user_defined_metadata_configs) == 0
            ):
                raise ValueError(
                    f"user_defined_metadata_configs in system output {x} has n't "
                    f"been specified or task name should be specified"
                )
            real_tasks.append(loader.user_defined_metadata_configs['task_name'])

    # Checks on other inputs
    # if num_outputs > 2:
    #     raise ValueError(
    #         f'ExplainaBoard currently only supports 1 or 2 system outputs,
    #         but received {num_outputs}'
    #     )
    if task is not None:
        if task not in TaskType.list():
            raise ValueError(
                f'Task name {task} was not recognized. ExplainaBoard currently '
                f'supports:{TaskType.list()}'
            )

    if file_type is not None:
        if file_type not in FileType.list():
            raise ValueError(
                f'File type {file_type} was not recognized. ExplainaBoard currently '
                f'supports: {FileType.list()}'
            )

    # Read in data and check validity
    if file_type is not None:
        if task is not None:
            loaders = [
                get_loader(task, data=x, file_type=file_type) for x in system_outputs
            ]
        elif len(real_tasks) > 0:
            loaders = [
                get_loader(task, data=x, file_type=file_type)
                for x, task in zip(system_outputs, real_tasks)
            ]
    else:
        loaders = [
            get_loader(task, data=x) for x in system_outputs
        ]  # use the default loaders that has been pre-defiend for each task
    system_datasets = [list(loader.load()) for loader in loaders]

    # validation
    if len(system_datasets) == 2:
        if len(system_datasets[0]) != len(system_datasets[1]):
            num0 = len(system_datasets[0])
            num1 = len(system_datasets[1])
            raise ValueError(
                f'Data must be identical for pairwise analysis, but length of files '
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
        "language": language,
        "task_name": task,
        "reload_stat": reload_stat,
        "conf_value": args.conf_value,
        "system_details": system_details,
    }

    if metric_names is not None:
        metadata["metric_names"] = metric_names

    # Run analysis
    reports = []
    for loader, system_dataset, system_full_path, task in zip(
        loaders, system_datasets, system_outputs, real_tasks
    ):

        metadata.update(loader.user_defined_metadata_configs)
        metadata.update(
            {"user_defined_features_configs": loader._user_defined_features_configs}
        )

        report = get_processor(task).process(
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

    if len(system_outputs) == 1:  # individual system analysis
        reports[0].print_as_json()
    elif len(system_outputs) == 2:  # pairwise analysis
        compare_analysis = get_pairwise_performance_gap(
            reports[0].to_dict(), reports[1].to_dict()
        )
        print(json.dumps(compare_analysis, indent=4))


if __name__ == '__main__':
    main()
