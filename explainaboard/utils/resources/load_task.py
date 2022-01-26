import json
import enum
from typing import List, Dict
import sys, os



PATH_OF_TASKS_JSON = "./tasks.json"


def get_task_mapping() -> Dict:
    with open(PATH_OF_TASKS_JSON,"r") as fin:
        task_infos = json.load(fin)
    return task_infos


def get_all_tasks() -> List[str]:
    task_infos = get_task_mapping()
    all_tasks = []
    for task_category, description in task_infos.items():
        task_list = description['options']
        all_tasks += task_list
    return all_tasks



all_tasks = get_all_tasks()
all_tasks_dict = {}
for task_name in all_tasks:
    all_tasks_dict[task_name.replace("-","_")] = task_name

TaskType = enum.Enum('TaskType', all_tasks_dict)


# TEST
# print(TaskType.summarization.value)


