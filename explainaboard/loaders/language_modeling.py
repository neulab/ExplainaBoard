"""Loaders for the language modeling task."""

from __future__ import annotations

from explainaboard.constants import FileType
from explainaboard.loaders.file_loader import (
    DatalabFileLoader,
    FileLoader,
    FileLoaderField,
    JSONFileLoader,
    TextFileLoader,
)
from explainaboard.loaders.loader import Loader


class LanguageModelingLoader(Loader):
    """Loader for the language modeling task.

    usage:
        please refer to `loaders_test.py`
    """

    @classmethod
    def default_dataset_file_type(cls) -> FileType:
        """See Loader.default_dataset_file_type."""
        return FileType.text

    @classmethod
    def default_dataset_file_loaders(cls) -> dict[FileType, FileLoader]:
        """See Loader.default_dataset_file_loaders."""
        field_name = "text"
        return {
            FileType.text: TextFileLoader(field_name, str),
            FileType.json: JSONFileLoader(
                [
                    FileLoaderField("text", field_name, str),
                ]
            ),
            FileType.datalab: DatalabFileLoader(
                [
                    FileLoaderField("text", field_name, str),  # text_column?
                ]
            ),
        }

    @classmethod
    def default_output_file_loaders(cls) -> dict[FileType, FileLoader]:
        """See Loader.default_output_file_loaders."""
        field_name = "log_probs"
        return {
            FileType.text: TextFileLoader(field_name, str),
            FileType.json: JSONFileLoader(
                [FileLoaderField(field_name, field_name, str)]
            ),
        }
