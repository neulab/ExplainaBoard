"""Loaders for the argument pair extraction class."""

from __future__ import annotations

from explainaboard.constants import FileType
from explainaboard.loaders.file_loader import (
    CoNLLFileLoader,
    DatalabFileLoader,
    FileLoader,
    FileLoaderField,
)
from explainaboard.loaders.loader import Loader


class ArgumentPairExtractionLoader(Loader):
    """A loader for argument pair extraction.

    usage:
        please refer to `loaders_test.py`
    """

    @classmethod
    def default_dataset_file_type(cls) -> FileType:
        """See Loader.default_dataset_file_type."""
        return FileType.datalab

    @classmethod
    def default_output_file_type(cls) -> FileType:
        """See Loader.default_output_file_type."""
        return FileType.conll

    @classmethod
    def default_dataset_file_loaders(cls) -> dict[FileType, FileLoader]:
        """See Loader.default_dataset_file_loaders."""
        field_names = ["sentences", "true_tags"]
        return {
            FileType.datalab: DatalabFileLoader(
                [
                    FileLoaderField("sentences_column", field_names[0], list),
                    FileLoaderField("labels_column", field_names[1], list),
                ]
            ),
        }

    @classmethod
    def default_output_file_loaders(cls) -> dict[FileType, FileLoader]:
        """See Loader.default_output_file_loaders."""
        return {
            FileType.conll: CoNLLFileLoader([FileLoaderField(1, "pred_tags", str)]),
        }
