from __future__ import annotations
from typing import Dict, Iterable, Optional
import typing as t
import json
from explainaboard.constants import FileType, Source
from explainaboard.loaders.file_loader import FileLoader
from explainaboard.tasks import TaskType

JSON = t.Union[str, int, float, bool, None, t.Mapping[str, 'JSON'], t.List['JSON']]  # type: ignore


class Loader:
    """Base class of Loaders

    Args:
        data: base64 encoded system output content or a path for the system output file
        source: source of data
        file type: tsv, json, conll, etc.
        file_loaders: a dict of file loaders. To customize the loading process, either implement
            a custome FileLoader or override `load()`
    """

    _default_source = Source.local_filesystem
    _default_file_type: Optional[FileType] = None
    _default_file_loaders: Dict[FileType, FileLoader] = {}

    def __init__(
        self,
        data: str,
        source: Optional[Source] = None,
        file_type: Optional[FileType] = None,
        file_loaders=None,
    ):
        if file_loaders is None:
            file_loaders = {}
        if not source and not self._default_source:
            raise Exception("no source is provided for the loader")
        if not file_type and not self._default_file_type:
            raise Exception("no file_type is provided for the loader")
        self._source = source or self._default_source
        self._file_type = file_type or self._default_file_type
        self.file_loaders = file_loaders or self._default_file_loaders
        self._data = data  # base64 or filepath

        if self._file_type not in self.file_loaders:
            raise NotImplementedError(
                f"A file loader for {self._file_type} is not provided. please add it to the file_loaders."
            )

        self._user_defined_features_configs: dict = (
            self._parse_user_defined_features_configs()
        )

    @property
    def user_defined_features_configs(self) -> dict:
        if self._user_defined_features_configs is None:
            raise Exception(
                "User defined features configs are not available (data has not been loaded))"
            )
        return self._user_defined_features_configs

    def _parse_user_defined_features_configs(self) -> dict:
        if self._file_type == FileType.json:
            raw_data = self.file_loaders[FileType.json].load_raw(
                self._data, self._source
            )
            if isinstance(raw_data, dict) and raw_data.get(
                "user_defined_features_configs"
            ):
                self._data = json.dumps(raw_data["predictions"])
                self._source = Source.in_memory
                return raw_data["user_defined_features_configs"]
        return {}

    def load(self) -> Iterable[dict]:
        file_loader = self.file_loaders[self._file_type]
        return file_loader.load(self._data, self._source)


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
