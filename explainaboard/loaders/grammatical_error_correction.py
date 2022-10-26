"""Loaders for grammatical error correction task."""

from __future__ import annotations

from explainaboard.constants import FileType
from explainaboard.loaders.file_loader import (
    DatalabFileLoader,
    FileLoader,
    FileLoaderField,
    JSONFileLoader,
)
from explainaboard.loaders.loader import Loader


class GrammaticalErrorCorrectionLoader(Loader):
    """Loader for the grammatical error correction task.

    usage:
        please refer to `loaders_test.py`
    """

    JSON_FIELDS: list[str | tuple[str, str]] = ["text", "edits"]

    @classmethod
    def default_dataset_file_type(cls) -> FileType:
        """See Loader.default_dataset_file_type."""
        return FileType.json

    @classmethod
    def default_output_file_type(cls) -> FileType:
        """See Loader.default_output_file_type."""
        return FileType.json

    @classmethod
    def default_dataset_file_loaders(cls) -> dict[FileType, FileLoader]:
        """See Loader.default_dataset_file_loaders."""
        return {
            FileType.json: JSONFileLoader(
                [
                    FileLoaderField("text", "text", str),
                    FileLoaderField("edits", "edits", dict),
                ]
            ),
            FileType.datalab: DatalabFileLoader(
                [
                    FileLoaderField("source_column", "text", str),
                    FileLoaderField("reference_column", "edits", dict),
                ]
            ),
        }

    @classmethod
    def default_output_file_loaders(cls) -> dict[FileType, FileLoader]:
        """See Loader.default_output_file_loaders."""
        field_name = "predicted_edits"
        return {
            FileType.json: JSONFileLoader(
                [FileLoaderField(field_name, field_name, dict)]
            ),
        }
