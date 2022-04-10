from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import json
from typing import Optional

from explainaboard.constants import FileType, Source
from explainaboard.loaders.file_loader import (
    FileLoader,
    FileLoaderField,
    TextFileLoader,
)
from explainaboard.tasks import TaskType
from explainaboard.utils.typing_utils import unwrap


class Loader:
    """Base class of Loaders

    :param data: base64 encoded system output content or a path for the system output
    file
    :param source: source of data
    :param file type: tsv, json, conll, etc.
    :param file_loaders: a dict of file loaders. To customize the loading process,
    either implement a custom FileLoader or override `load()`
    """

    @classmethod
    def default_source(cls) -> Source:
        return Source.local_filesystem

    @classmethod
    def default_file_type(cls) -> Optional[FileType]:
        return None

    @classmethod
    def default_dataset_file_loaders(cls) -> dict[FileType, FileLoader]:
        return {}

    @classmethod
    def default_output_file_loaders(cls) -> FileLoader:
        return TextFileLoader("output", str)

    def __init__(
        self,
        data: str,
        source: Optional[Source] = None,
        file_type: Optional[FileType] = None,
        file_loader: FileLoader = None,
    ):
        if not source and not self.default_source():
            raise ValueError("no source is provided for the loader")
        else:
            self._source: Source = source or self.default_source()

        if not file_type and not self.default_file_type():
            raise ValueError("no file_type is provided for the loader")
        else:
            self._file_type = file_type or unwrap(self.default_file_type())

        if file_loader:
            self.file_loader: FileLoader = file_loader
        else:
            default_file_loaders = self.default_dataset_file_loaders()
            if self._file_type not in default_file_loaders:
                raise ValueError(
                    f"A file loader for {self._file_type} is not provided. "
                    "please pass it in as an argument."
                )
            else:
                self.file_loader = default_file_loaders[self._file_type]

        self._data = data  # base64 or filepath
        self._user_defined_features_configs: dict[
            str, CustomFeature
        ] = self._parse_user_defined_features_configs()

        self._user_defined_metadata_configs: dict = (
            self._parse_user_defined_metadata_configs()
        )

    @property
    def user_defined_features_configs(self) -> dict[str, CustomFeature]:
        if self._user_defined_features_configs is None:
            raise Exception(
                "User defined features configs are not available "
                "(data has not been loaded))"
            )
        return self._user_defined_features_configs

    @property
    def user_defined_metadata_configs(self) -> dict:
        if self._user_defined_metadata_configs is None:
            raise Exception(
                "User defined metadata configs are not available "
                "(data has not been loaded))"
            )
        return self._user_defined_metadata_configs

    def _parse_user_defined_features_configs(self) -> dict:
        """custom features can only be defined in JSON files"""
        if self._file_type == FileType.json:
            raw_data = self.file_loader.load_raw(self._data, self._source)
            if isinstance(raw_data, dict) and raw_data.get(
                "user_defined_features_configs"
            ):
                self._data = json.dumps(raw_data["predictions"])
                self._source = Source.in_memory
                custom_feature_configs: dict[str, CustomFeature] = {
                    name: CustomFeature.from_dict(name, dikt)
                    for name, dikt in raw_data["user_defined_features_configs"].items()
                }
                fields: list[FileLoaderField] = []
                for feature in custom_feature_configs.values():
                    fields.append(
                        # dtype is set to None because custom feature configs doesn't
                        # use the same set of dtypes as FileLoader (this is not
                        # enforced anywhere)
                        FileLoaderField(feature.name, feature.name, None, False)
                    )
                self.file_loader.add_fields(fields)
                return custom_feature_configs
        return {}

    def _parse_user_defined_metadata_configs(self) -> dict:
        if self._file_type == FileType.json:
            raw_data = self.file_loader.load_raw(self._data, self._source)
            if isinstance(raw_data, dict) and raw_data.get(
                "user_defined_metadata_configs"
            ):
                self._data = json.dumps(raw_data["predictions"])
                self._source = Source.in_memory
                return raw_data["user_defined_metadata_configs"]
        return {}

    def load(self) -> Iterable[dict]:
        return self.file_loader.load(self._data, self._source)


# loader_registry is a global variable, storing all basic loading functions
_loader_registry: dict[TaskType, type[Loader]] = {}


def get_loader(
    task: TaskType | str,
    data: str,
    source: Source | None = None,
    file_type: FileType | str | None = None,
) -> Loader:
    task_cast: TaskType = TaskType(task)
    file_type_cast: FileType | None = (
        FileType(file_type) if file_type is not None else None
    )
    return _loader_registry[task_cast](data, source, file_type_cast)


def register_loader(task_type: TaskType):
    """
    a register for different data loaders, for example
    For example, `@register_loader(TaskType.text_classification)`
    """

    def register_loader_fn(cls):
        _loader_registry[task_type] = cls
        return cls

    return register_loader_fn


@dataclass
class CustomFeature:
    name: str
    dtype: str
    description: str
    num_buckets: int

    @classmethod
    def from_dict(cls, name: str, dikt: dict) -> CustomFeature:
        return CustomFeature(
            name,
            dtype=dikt["dtype"],
            description=dikt["description"],
            num_buckets=dikt["num_buckets"],
        )
