import argparse
import json

from explainaboard import (
    get_loader,
    get_pairwise_performance_gap,
    get_processor,
    TaskType,
)


def main():

    parser = argparse.ArgumentParser(description='Explainable Leaderboards for NLP')

    parser.add_argument('--task', type=str, required=True, help="the task name")

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
        '--conf_value',
        type=float,
        required=False,
        default=0.05,
        help="the p-value with which to calculate the confidence interval",
    )

    args = parser.parse_args()

    dataset = args.dataset
    sub_dataset = args.sub_dataset
    task = args.task
    reload_stat = False if args.reload_stat == "0" else True
    system_outputs = args.system_outputs
    num_outputs = len(system_outputs)
    metric_names = args.metrics

    # Checks on inputs
    if num_outputs > 2:
        raise ValueError(
            f'ExplainaBoard currently only supports 1 or 2 system outputs, but '
            f'received {num_outputs}'
        )
    if task not in TaskType.list():
        raise ValueError(
            f'Task name {task} was not recognized. ExplainaBoard currently supports: '
            f'{TaskType.list()}'
        )

    # Read in data and check validity
    loaders = [get_loader(task, data=x) for x in system_outputs]
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
        "task_name": task,
        "reload_stat": reload_stat,
        "user_defined_features_configs": loaders[0].user_defined_features_configs,
        "conf_value": args.conf_value,
    }
    if metric_names is not None:
        metadata["metric_names"] = metric_names

    # Run analysis
    reports = [
        get_processor(task).process(metadata=metadata, sys_output=x)
        for x in system_datasets
    ]
    if len(system_outputs) == 1:  # individual system analysis
        reports[0].print_as_json()
    else:  # pairwise analysis
        compare_analysis = get_pairwise_performance_gap(
            reports[0].to_dict(), reports[1].to_dict()
        )
        print(json.dumps(compare_analysis, indent=4))


if __name__ == '__main__':
    main()
