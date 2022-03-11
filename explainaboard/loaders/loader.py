from typing import Dict, Iterable, List
import typing as t
import json
from enum import Enum
from io import StringIO
import csv
from explainaboard.constants import FileType, Source
from explainaboard.tasks import TaskType

JSON = t.Union[str, int, float, bool, None, t.Mapping[str, 'JSON'], t.List['JSON']]  # type: ignore


class Loader:
    """base class of loader"""

    def __init__(self, source: Source, file_type: Enum, data: str):
        self._source = source
        self._file_type = file_type
        self._data = data

    def load_user_defined_features_configs(self):
        return {}

    def _load_raw_data_points(self) -> Iterable:
        """
        loads data and return an iterable of data points. element type depends on file_type
        TODO: error handling
        """
        if self._source == Source.in_memory:
            if self._file_type == FileType.tsv:
                file = StringIO(self._data)
                return csv.reader(file, delimiter='\t')
            elif self._file_type == FileType.conll:
                return self._data.splitlines()
            elif self._file_type == FileType.json:
                return json.loads(self._data)
            elif self._file_type == FileType.datalab:
                return self._data
            else:
                raise NotImplementedError

        elif self._source == Source.local_filesystem:
            if self._file_type == FileType.tsv:
                content: List[str] = []
                with open(self._data, "r", encoding="utf8") as fin:
                    for record in csv.reader(fin, delimiter='\t'):
                        content.append(record)
                return content
            elif self._file_type == FileType.conll:
                content = []
                with open(self._data, "r", encoding="utf8") as fin:
                    for record in fin:
                        content.append(record)
                return content
            elif self._file_type == FileType.json:
                with open(self._data, 'r', encoding="utf8") as json_file:
                    data = json_file.read()
                obj = json.loads(data)
                return obj
            else:
                raise NotImplementedError

    def load(self) -> Iterable[Dict]:
        raise NotImplementedError


# loader_registry is a global variable, storing all basic loading functions
_loader_registry: Dict = {}


def get_loader(
    task: TaskType, source: Source = None, file_type: FileType = None, data: str = None
) -> Loader:

    return _loader_registry[task](source, file_type, data)


def register_loader(task_type: TaskType):
    """
    a register for different data loaders, for example
    For example, `@register_loader(TaskType.text_classification)`
    """

    def register_loader_fn(cls):
        _loader_registry[task_type] = cls
        return cls

    return register_loader_fn
