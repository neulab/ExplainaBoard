from typing import List
from explainaboard import feature
from explainaboard.tasks import TaskType
from explainaboard.info import SysOutputInfo
from explainaboard.builders import ExplainaboardBuilder


class Processor:
    """Base case for task-based processor"""

    _features: feature.Features
    _task_type: TaskType
    # TODO(gneubig): this could potentially be moved directly into the task definition
    _default_metrics: List[str]
    _builder: ExplainaboardBuilder

    def __init__(self) -> None:
        pass

    def process(self, metadata: dict, sys_output: List[dict]) -> SysOutputInfo:
        if metadata is None:
            metadata = {}
        if "task_name" not in metadata.keys():
            metadata["task_name"] = self._task_type.value
        if "metric_names" not in metadata.keys():
            metadata["metric_names"] = self._default_metrics
        sys_info = SysOutputInfo.from_dict(metadata)
        sys_info.features = self._features
        sys_info.results = self._builder.run(sys_info, sys_output)
        return sys_info
