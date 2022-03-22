from typing import Dict, Iterable, List, Optional
import typing as t
import json
from io import StringIO
import csv
from explainaboard.constants import FileType, Source
from explainaboard.tasks import TaskType

JSON = t.Union[str, int, float, bool, None, t.Mapping[str, 'JSON'], t.List['JSON']]  # type: ignore


class Loader:
    """base class of loader"""

    _default_source = Source.local_filesystem
    _default_file_type: Optional[FileType] = None

    def __init__(
        self,
        data: str,
        source: Optional[Source] = None,
        file_type: Optional[FileType] = None,
    ):
        if not source and not self._default_source:
            raise Exception("no source is provided for the loader")
        if not file_type and not self._file_type:
            raise Exception("no file_type is provided for the loader")
        self._source = source or self._default_source
        self._file_type = file_type or self._default_file_type
        self._data = data  # base64 or filepath
        self._raw_data: Optional[Iterable] = None  # loaded data

        # None: uninitialized; {}: no custom features defined
        self._user_defined_features_configs: Optional[dict] = None

    @property
    def user_defined_features_configs(self) -> dict:
        if self._user_defined_features_configs is None:
            raise Exception(
                "User defined features configs are not available (data has not been loaded))"
            )
        return self._user_defined_features_configs

    def _load_raw_data_points(self) -> Iterable:
        """
        loads data and return an iterable of data points. element type depends on file_type
        """
        raw_data: Optional[Iterable] = None
        if self._source == Source.in_memory:
            if self._file_type == FileType.tsv:
                file = StringIO(self._data)
                raw_data = csv.reader(file, delimiter='\t')
            elif self._file_type == FileType.conll:
                raw_data = self._data.splitlines()
            elif self._file_type == FileType.json:
                raw_data = json.loads(self._data)
            elif self._file_type == FileType.datalab:
                raw_data = self._data
            else:
                raise NotImplementedError

        elif self._source == Source.local_filesystem:
            if self._file_type == FileType.tsv:
                content: List[str] = []
                with open(self._data, "r", encoding="utf8") as fin:
                    for record in csv.reader(fin, delimiter='\t'):
                        content.append(record)
                raw_data = content
            elif self._file_type == FileType.conll:
                content = []
                with open(self._data, "r", encoding="utf8") as fin:
                    for record in fin:
                        content.append(record)
                raw_data = content
            elif self._file_type == FileType.json:
                with open(self._data, 'r', encoding="utf8") as json_file:
                    data = json_file.read()
                raw_data = json.loads(data)
            else:
                raise NotImplementedError

        # load user defined features if exists
        if isinstance(raw_data, dict) and raw_data.get("user_defined_features_configs"):
            self._user_defined_features_configs = raw_data[
                "user_defined_features_configs"
            ]
            raw_data = raw_data["predictions"]

        else:
            self._user_defined_features_configs = {}
        self._raw_data = raw_data
        return raw_data

    def load(self) -> Iterable[dict]:
        return self._load_raw_data_points()


# loader_registry is a global variable, storing all basic loading functions
_loader_registry: Dict = {}


def get_loader(
    task: TaskType,
    data: str,
    source: Optional[Source] = None,
    file_type: Optional[FileType] = None,
) -> Loader:

    return _loader_registry[task](data, source, file_type)


def register_loader(task_type: TaskType):
    """
    a register for different data loaders, for example
    For example, `@register_loader(TaskType.text_classification)`
    """

    def register_loader_fn(cls):
        _loader_registry[task_type] = cls
        return cls

    return register_loader_fn
