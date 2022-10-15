"""Loaders for the extractive QA task."""

from __future__ import annotations

from explainaboard.constants import FileType
from explainaboard.loaders.file_loader import (
    DatalabFileLoader,
    FileLoader,
    FileLoaderField,
    JSONFileLoader,
)
from explainaboard.loaders.loader import Loader


class QAExtractiveLoader(Loader):
    """Loader for the extractive QA task."""

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
        target_field_names = ["context", "question", "answers"]
        return {
            FileType.json: JSONFileLoader(
                [
                    FileLoaderField(
                        target_field_names[0],
                        target_field_names[0],
                        str,
                        strip_before_parsing=False,
                    ),
                    FileLoaderField(
                        target_field_names[1],
                        target_field_names[1],
                        str,
                        strip_before_parsing=False,
                    ),
                    FileLoaderField(target_field_names[2], target_field_names[2]),
                ]
            ),
            FileType.datalab: DatalabFileLoader(
                [
                    FileLoaderField(
                        "context_column",
                        target_field_names[0],
                        str,
                        strip_before_parsing=False,
                    ),
                    FileLoaderField(
                        "question_column",
                        target_field_names[1],
                        str,
                        strip_before_parsing=False,
                    ),
                    FileLoaderField("answers_column", target_field_names[2]),
                ]
            ),
        }

    @classmethod
    def default_output_file_loaders(cls) -> dict[FileType, FileLoader]:
        """See Loader.default_output_file_loaders."""
        return {
            FileType.json: JSONFileLoader(
                [FileLoaderField("predicted_answers", "predicted_answers")]
            )
        }
