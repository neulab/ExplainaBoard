from __future__ import annotations

from dataclasses import dataclass
from typing import final, Literal, Optional

from explainaboard.constants import FileType, Source
from explainaboard.loaders.file_loader import (
    DatalabLoaderOption,
    FileLoader,
    FileLoaderReturn,
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
        dataset_source: Source | None = None,
        output_source: Source | None = None,
        dataset_file_type: FileType | None = None,
        output_file_type: FileType | None = None,
        dataset_file_loader: FileLoader | None = None,
        output_file_loader: FileLoader | None = None,
        field_mapping: dict[str, str] | None = None,
    ):
        # determine sources
        self._dataset_source: Source = dataset_source or self.default_source()
        self._output_source: Source = output_source or self.default_source()

        # save field mapping
        self._field_mapping: dict[str, str] = field_mapping or {}

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

    def load(self) -> FileLoaderReturn:
        dataset_loaded_data = self._dataset_file_loader.load(
            self._dataset_data, self._dataset_source, field_mapping=self._field_mapping
        )
        output_loaded_data = self._output_file_loader.load(
            self._output_data, self._output_source, field_mapping=self._field_mapping
        )
        dataset_loaded_data.metadata.merge(output_loaded_data.metadata)
        if len(dataset_loaded_data) != len(output_loaded_data):
            raise ValueError(
                "dataset and output are of different length"
                + f"({len(dataset_loaded_data)} != {len(output_loaded_data)})"
            )
        data_list: list[dict] = output_loaded_data.samples
        for i, output in enumerate(data_list):
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
