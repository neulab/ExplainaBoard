"""Load information about a task mapping."""

from __future__ import annotations

import enum
import json

PATH_OF_TASKS_JSON = "./tasks.json"


def get_task_mapping() -> dict:
    """Get the mapping between tasks.

    Returns:
        A dictionary specifying the task mapping.
    """
    with open(PATH_OF_TASKS_JSON, "r") as fin:
        task_infos = json.load(fin)
    return task_infos


def get_all_tasks() -> list[str]:
    """Get all tasks that are available.

    Returns:
        A list of tasks.
    """
    task_infos = get_task_mapping()
    all_tasks = []
    for task_category, description in task_infos.items():
        task_list = description["options"]
        all_tasks += task_list
    return all_tasks


all_tasks = get_all_tasks()
all_tasks_dict = {}
for task_name in all_tasks:
    all_tasks_dict[task_name.replace("-", "_")] = task_name

# TODO(odashi): avoid storing list of tasks into enum as num should be considered
# as a static type.
TaskType = enum.Enum("TaskType", all_tasks_dict)  # type: ignore
