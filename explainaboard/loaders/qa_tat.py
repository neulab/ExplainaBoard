"""Loaders for the TAT-QA dataset."""

from __future__ import annotations

from explainaboard.constants import FileType
from explainaboard.loaders.file_loader import (
    DatalabFileLoader,
    FileLoader,
    FileLoaderField,
    JSONFileLoader,
)
from explainaboard.loaders.loader import Loader


class QATatLoader(Loader):
    """Loader for the TAT-QA class."""

    @classmethod
    def default_dataset_file_type(cls) -> FileType:
        """See Loader.default_dataset_file_type."""
        return FileType.json

    @classmethod
    def default_output_file_type(cls) -> FileType:
        """See Loader.default_output_file_type."""
        return FileType.text

    @classmethod
    def default_dataset_file_loaders(cls) -> dict[FileType, FileLoader]:
        """See Loader.default_dataset_file_loaders."""
        return {
            FileType.datalab: DatalabFileLoader(
                [
                    FileLoaderField("question_column", "question", str),
                    FileLoaderField("context_column", "context", list),
                    FileLoaderField("table_column", "table", list),
                    FileLoaderField("answer_column", "true_answer", list),
                    FileLoaderField("answer_type_column", "answer_type", str),
                    FileLoaderField("answer_scale_column", "answer_scale", str),
                ]
            ),
        }

    @classmethod
    def default_output_file_loaders(cls) -> dict[FileType, FileLoader]:
        """See Loader.default_output_file_loaders."""
        return {
            FileType.json: JSONFileLoader(
                [
                    FileLoaderField("answer", "predicted_answer", list),
                    FileLoaderField("scale", "predicted_answer_scale", str),
                ]
            )
        }
