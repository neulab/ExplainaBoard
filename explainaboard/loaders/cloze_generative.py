"""Loaders for the generative close task."""

from __future__ import annotations

from explainaboard.constants import FileType
from explainaboard.loaders.file_loader import (
    DatalabFileLoader,
    FileLoader,
    FileLoaderField,
    JSONFileLoader,
)
from explainaboard.loaders.loader import Loader


class ClozeGenerativeLoader(Loader):
    """Loader for the generative cloze task."""

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
        target_field_names = ["context", "hint", "question_mark", "answers"]
        return {
            FileType.json: JSONFileLoader(
                [
                    FileLoaderField("context", target_field_names[0], str),
                    FileLoaderField("hint", target_field_names[1], str),
                    FileLoaderField("question_mark", target_field_names[2], str),
                    FileLoaderField("answers", target_field_names[3], list),
                ]
            ),
            FileType.datalab: DatalabFileLoader(
                [
                    FileLoaderField("context_column", target_field_names[0], str),
                    FileLoaderField("hint_column", target_field_names[1], str),
                    FileLoaderField("question_column", target_field_names[2], str),
                    FileLoaderField("answers_column", target_field_names[3], list),
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
