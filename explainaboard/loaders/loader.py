from __future__ import annotations

from dataclasses import dataclass, field

from explainaboard.constants import FileType, Source
from explainaboard.loaders.file_loader import (
    DatalabLoaderOption,
    FileLoader,
    FileLoaderReturn,
    TextFileLoader,
)
from explainaboard.utils.typing_utils import unwrap, unwrap_or_else


@dataclass
class SupportedFileTypes:
    """List of dataset/output file types supported by the loader."""

    custom_dataset: list[FileType] = field(default_factory=list)
    system_output: list[FileType] = field(default_factory=list)


class Loader:
    """Base class of Loaders
    - system output is split into two parts: the dataset (features and true labels) and
    the output (predicted labels)
    """

    @classmethod
    def from_datalab(
        cls,
        dataset: DatalabLoaderOption,
        output_data: str | None,
        output_source: Source | None = None,
        output_file_type: FileType | None = None,
        field_mapping: dict[str, str] | None = None,
    ) -> Loader:
        """Convenient method to initializes a loader for a dataset from datalab.

        The loader downloads the dataset and merges the user provided output with the
        dataset.
        """
        return cls(
            dataset_data=dataset,
            output_data=output_data,
            dataset_source=Source.in_memory,
            output_source=output_source,
            dataset_file_type=FileType.datalab,
            output_file_type=output_file_type,
            field_mapping=field_mapping,
        )

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

    @classmethod
    def supported_file_types(cls) -> SupportedFileTypes:
        return SupportedFileTypes(
            list(cls.default_dataset_file_loaders().keys()),
            list(cls.default_output_file_loaders().keys()),
        )

    def __init__(
        self,
        dataset_data: str | DatalabLoaderOption,
        output_data: str | None,
        dataset_source: Source | None = None,
        output_source: Source | None = None,
        dataset_file_type: FileType | None = None,
        output_file_type: FileType | None = None,
        dataset_file_loader: FileLoader | None = None,
        output_file_loader: FileLoader | None = None,
        field_mapping: dict[str, str] | None = None,
    ):
        """
        Class initializer for the loader
        :param dataset_data: a string pointing to a dataset file, or a specification of
          the location of the data within DataLab
        :param output_data: a string pointing to a system output file, or `None`, if
          no output file will be read in
        :param dataset_source: source of the dataset (e.g. local disk or network)
        :param output_source: source of the system output
        :param dataset_file_type: dataset file type (e.g. tsv, json, conll, etc.)
        :param output_file_type: system output file type
        :param dataset_file_loader: a FileLoader to customize the dataset loading
          process
        :param output_file_loader: FileLoader for output files
        :param field_mapping: A mapping from fields in the original data to fields in
          the output data
        """

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
        try:
            self._dataset_file_loader = unwrap_or_else(
                dataset_file_loader,
                lambda: self.default_dataset_file_loaders()[unwrap(dataset_file_type)],
            )
        except KeyError:
            raise ValueError(
                f"{dataset_file_type} is not a supported dataset file type of "
                f"{self.__class__.__name__}."
            )
        try:
            self._output_file_loader = unwrap_or_else(
                output_file_loader,
                lambda: self.default_output_file_loaders()[unwrap(output_file_type)],
            )
        except KeyError:
            raise ValueError(
                f"{output_file_type} is not a supported output file type of "
                f"{self.__class__.__name__}."
            )

        self._dataset_data = dataset_data  # base64, filepath or datalab options
        self._output_data = output_data

    def load(self) -> FileLoaderReturn:
        """
        Load data from the dataset and output.
        If `output_data` is `None`, then data will only be returned from the dataset.
        :returns: A FileLoaderReturn object with samples and metadata.
        """

        dataset_loaded_data = self._dataset_file_loader.load(
            self._dataset_data, self._dataset_source, field_mapping=self._field_mapping
        )
        if self._output_data:
            output_loaded_data = self._output_file_loader.load(
                self._output_data,
                self._output_source,
                field_mapping=self._field_mapping,
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
