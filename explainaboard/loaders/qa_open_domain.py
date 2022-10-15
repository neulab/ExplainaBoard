"""Loaders for the open domain QA task."""

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


class QAOpenDomainLoader(Loader):
    """Loader for the open domain QA task."""

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
        target_field_names = ["question", "answers"]
        return {
            FileType.json: JSONFileLoader(
                [
                    FileLoaderField("question", target_field_names[0], str),
                    FileLoaderField("answers", target_field_names[1], list),
                ]
            ),
            FileType.datalab: DatalabFileLoader(
                [
                    FileLoaderField("question_column", target_field_names[0], str),
                    FileLoaderField("answers_column", target_field_names[1], list),
                ]
            ),
        }

    @classmethod
    def default_output_file_loaders(cls) -> dict[FileType, FileLoader]:
        """See Loader.default_output_file_loaders."""
        field_name = "predicted_answer"
        return {
            FileType.text: TextFileLoader(field_name, str),
        }
