"""Loaders for the text-to-SQL task."""
from __future__ import annotations

from explainaboard.constants import FileType
from explainaboard.loaders.file_loader import (
    FileLoader,
    FileLoaderField,
    JSONFileLoader,
    TextFileLoader,
)
from explainaboard.loaders.loader import Loader


class TextToSQLLoader(Loader):
    """Loader for the text_to_SQL task.

    usage:
        please refer to `loaders_test.py`
    """

    @classmethod
    def default_dataset_file_type(cls) -> FileType:
        """See Loader.default_dataset_file_type."""
        return FileType.json

    @classmethod
    def default_dataset_file_loaders(cls) -> dict[FileType, FileLoader]:
        """See Loader.default_dataset_file_loaders."""
        return {
            FileType.json: JSONFileLoader(
                [
                    FileLoaderField("question", "question", str),
                    FileLoaderField("query", "true_sql", str),
                    FileLoaderField("db_id", "db_id", str),
                ]
            ),
        }

    @classmethod
    def default_output_file_loaders(cls) -> dict[FileType, FileLoader]:
        """See Loader.default_output_file_loaders."""
        return {
            FileType.text: TextFileLoader("predicted_sql", str),
        }
