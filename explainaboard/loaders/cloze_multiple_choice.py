"""Loaders for the multiple choice cloze task."""

from __future__ import annotations

from explainaboard.constants import FileType
from explainaboard.loaders.file_loader import (
    DatalabFileLoader,
    FileLoader,
    FileLoaderField,
    JSONFileLoader,
)
from explainaboard.loaders.loader import Loader


class ClozeMultipleChoiceLoader(Loader):
    """Loader for the multiple choice cloze task."""

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
        target_field_names = ["context", "options", "question_mark", "answers"]
        return {
            FileType.json: JSONFileLoader(
                [
                    FileLoaderField("context", target_field_names[0], str),
                    FileLoaderField("options", target_field_names[1], list),
                    FileLoaderField("question_mark", target_field_names[2], str),
                    FileLoaderField("answers", target_field_names[3], dict),
                ]
            ),
            FileType.datalab: DatalabFileLoader(
                [
                    FileLoaderField("context_column", target_field_names[0], str),
                    FileLoaderField("options_column", target_field_names[1], list),
                    FileLoaderField("question_column", target_field_names[2], str),
                    FileLoaderField("answers_column", target_field_names[3], dict),
                ]
            ),
        }

    @classmethod
    def default_output_file_loaders(cls) -> dict[FileType, FileLoader]:
        """See Loader.default_output_file_loaders."""
        return {
            FileType.json: JSONFileLoader(
                [FileLoaderField("predicted_answers", "predicted_answers", dict)]
            )
        }
