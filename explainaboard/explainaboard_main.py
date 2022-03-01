import argparse
import json
from explainaboard import get_loader, get_processor
from explainaboard import TaskType


def get_performance_gap(sys1, sys2):

    for metric_name, performance_unit in sys1["results"]["overall"].items():
        sys1["results"]["overall"][metric_name]["value"] = float(
            sys1["results"]["overall"][metric_name]["value"]
        ) - float(sys2["results"]["overall"][metric_name]["value"])
        sys1["results"]["overall"][metric_name]["confidence_score_low"] = 0
        sys1["results"]["overall"][metric_name]["confidence_score_up"] = 0

    for attr, performance_list in sys1["results"]["fine_grained"].items():
        for idx, performances in enumerate(performance_list):
            for idy, performance_unit in enumerate(
                performances
            ):  # multiple metrics' results
                sys1["results"]["fine_grained"][attr][idx][idy]["value"] = float(
                    sys1["results"]["fine_grained"][attr][idx][idy]["value"]
                ) - float(sys2["results"]["fine_grained"][attr][idx][idy]["value"])
                sys1["results"]["fine_grained"][attr][idx][idy][
                    "confidence_score_low"
                ] = 0
                sys1["results"]["fine_grained"][attr][idx][idy][
                    "confidence_score_up"
                ] = 0

    return sys1


def main():

    parser = argparse.ArgumentParser(description='Explainable Leaderboards for NLP')

    parser.add_argument('--task', type=str, required=True, help="the task name")

    parser.add_argument(
        '--system_outputs',
        type=str,
        required=True,
        nargs="+",
        help="the directories of system outputs. Multiple one should be separated by space, for example: system1 system2",
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
        '--metrics',
        type=str,
        required=False,
        nargs="*",
        help="multiple metrics should be separated by space",
    )

    args = parser.parse_args()

    dataset = args.dataset
    task = args.task
    system_outputs = args.system_outputs
    num_outputs = len(system_outputs)
    metric_names = args.metrics

    # Checks on inputs
    if num_outputs > 2:
        raise ValueError(
            f'ExplainaBoard currently only supports 1 or 2 system outputs, but received {num_outputs}'
        )
    if task not in TaskType.list():
        raise ValueError(
            f'Task name {task} was not recognized. ExplainaBoard currently supports: {TaskType.list()}'
        )

    # Read in data and check validity
    system_datasets = [get_loader(task, data=x).load() for x in system_outputs]
    if len(system_datasets) == 2 and len(system_datasets[0]) != len(system_datasets[1]):
        num0 = len(system_datasets[0])
        num1 = len(system_datasets[1])
        raise ValueError(
            f'Data must be identical for pairwise analysis, but length of files {system_datasets[0]} ({num0}) != {system_datasets[1]} ({num1})'
        )

    # Setup metadata
    metadata = {"dataset_name": dataset, "task_name": task}
    if metric_names is not None:
        metadata["metric_names"] = metric_names

    # Run analysis
    reports = [get_processor(task, metadata=metadata, data=x).process() for x in system_datasets]
    if len(system_outputs) == 1:  # individual system analysis
        reports[0].print_as_json()
    else:                         # pairwise analysis
        compare_analysis = get_performance_gap(reports[0].to_dict(), reports[1].to_dict())
        print(json.dumps(compare_analysis, indent=4))


if __name__ == '__main__':
    main()
