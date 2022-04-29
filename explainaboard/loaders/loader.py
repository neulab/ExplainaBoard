from __future__ import annotations

from dataclasses import dataclass
import json
from typing import final, Literal, Optional

from explainaboard.constants import FileType, Source
from explainaboard.loaders.file_loader import (
    DatalabLoaderOption,
    FileLoader,
    FileLoaderField,
    JSONFileLoader,
    TextFileLoader,
)


class Loader:
    """Base class of Loaders
    - system output is split into two parts: the dataset (features and true labels) and
    the output (predicted labels)
    :param data: if str, base64 encoded system output or a path.
    :param source: source of data
    :param file_type: tsv, json, conll, etc.
    :param file_loader: a dict of file loaders. To customize the loading process,
    either implement a custom FileLoader or override `load()`
    """

    @classmethod
    def default_source(cls) -> Source:
        return Source.local_filesystem

    @classmethod
    def default_dataset_file_type(cls) -> FileType:
        return FileType.json

    @classmethod
    def default_output_file_type(cls) -> FileType:
        return FileType.text

    @classmethod
    def default_dataset_file_loaders(cls) -> dict[FileType, FileLoader]:
        return {}

    @classmethod
    def default_output_file_loaders(cls) -> dict[FileType, FileLoader]:
        return {FileType.text: TextFileLoader()}

    def __init__(
        self,
        dataset_data: str | DatalabLoaderOption,
        output_data: str,
        dataset_source: Optional[Source] = None,
        output_source: Optional[Source] = None,
        dataset_file_type: Optional[FileType] = None,
        output_file_type: Optional[FileType] = None,
        dataset_file_loader: Optional[FileLoader] = None,
        output_file_loader: Optional[FileLoader] = None,
    ):
        # determine sources
        self._dataset_source: Source = dataset_source or self.default_source()
        self._output_source: Source = output_source or self.default_source()

        # determine file types
        if not dataset_file_type:
            dataset_file_type = self.default_dataset_file_type()
        if not output_file_type:
            output_file_type = self.default_output_file_type()

        # determine file loaders
        self._dataset_file_loader = self.select_file_loader(
            "dataset", dataset_file_type, dataset_file_loader
        )
        self._output_file_loader = self.select_file_loader(
            "output", output_file_type, output_file_loader
        )

        self._dataset_data = dataset_data  # base64, filepath or datalab options
        self._output_data = output_data

        self._user_defined_features_configs: dict[str, CustomFeature] = {}
        self._user_defined_metadata_configs: dict = {}
        if output_file_type == FileType.json:
            (
                self._user_defined_features_configs,
                self._user_defined_metadata_configs,
            ) = self._parse_user_defined_fields()

    @classmethod
    @final
    def select_file_loader(
        cls,
        split: Literal["dataset", "output"],
        file_type: FileType,
        custom_loader: Optional[FileLoader],
    ) -> FileLoader:
        if custom_loader:
            return custom_loader
        else:
            if split == "dataset":
                default_file_loaders = cls.default_dataset_file_loaders()
            elif split == "output":
                default_file_loaders = cls.default_output_file_loaders()
            else:
                raise ValueError("split must be one of [dataset, output]")
            if file_type not in default_file_loaders:
                raise ValueError(
                    f"A file loader for {file_type} is not provided. "
                    "please pass it in as an argument."
                )
            else:
                return default_file_loaders[file_type]

    @property
    def user_defined_features_configs(self) -> dict[str, CustomFeature]:
        return self._user_defined_features_configs

    @property
    def user_defined_metadata_configs(self) -> dict:
        return self._user_defined_metadata_configs

    def _parse_user_defined_fields(self) -> tuple[dict[str, CustomFeature], dict]:
        """custom features and metadata can only be defined in the output file and it
        needs to be in JSON format"""
        if isinstance(self._output_file_loader, JSONFileLoader):
            raw_data = self._output_file_loader.load_raw(
                self._output_data, self._output_source
            )
            if isinstance(raw_data, dict):
                custom_features = raw_data.get("user_defined_features_configs", {})
                metadata = raw_data.get("user_defined_metadata_configs", {})

                if custom_features or metadata:
                    if "predictions" not in raw_data:
                        raise ValueError(
                            "system output file is missing predictions field"
                        )
                    # replace output data with the predictions only
                    self._output_data = json.dumps(raw_data["predictions"])
                    self._output_source = Source.in_memory

                custom_feature_configs: dict[str, CustomFeature] = {}
                if custom_features:  # add custom features to output file loader fields
                    custom_feature_configs = {
                        name: CustomFeature.from_dict(name, dikt)
                        for name, dikt in raw_data[
                            "user_defined_features_configs"
                        ].items()
                    }
                    fields: list[FileLoaderField] = []
                    for feature in custom_feature_configs.values():
                        fields.append(
                            # dtype is set to None because custom feature configs
                            # doesn't use the same set of dtypes as FileLoader
                            # (this is not enforced anywhere)
                            FileLoaderField(feature.name, feature.name, None, False)
                        )
                    self._output_file_loader.add_fields(fields)
                return custom_feature_configs, metadata
            return {}, {}
        else:
            raise Exception(
                "_parse_user_defined_fields can only be called with a JSON system "
                + "output file"
            )

    def load(self) -> list[dict]:
        dataset_loaded_data = self._dataset_file_loader.load(
            self._dataset_data, self._dataset_source
        )
        output_loaded_data = self._output_file_loader.load(
            self._output_data, self._output_source
        )
        if len(dataset_loaded_data) != len(output_loaded_data):
            raise ValueError(
                "dataset and output are of different length"
                + f"({len(dataset_loaded_data)} != {len(output_loaded_data)})"
            )
        for i, output in enumerate(output_loaded_data):
            dataset_loaded_data[i].update(output)
        return dataset_loaded_data


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
