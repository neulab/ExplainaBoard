from typing import Any, Iterable, Optional
from explainaboard import feature
from explainaboard.tasks import TaskType
from explainaboard.info import SysOutputInfo


class Processor:
    """Base case for task-based processor"""
    _features: feature.Features
    _task_type: TaskType

    def __init__(self, metadata: dict, system_output_data: Iterable[dict]) -> None:
        self._metadata = {**metadata, "features": self._features}
        self._system_output_info = SysOutputInfo.from_dict(self._metadata)
        # should really be a base type of builders
        self._builder: Optional[Any] = None

    def process(self) -> SysOutputInfo:
        if not self._builder:
            raise NotImplementedError
        return self._builder.run()
