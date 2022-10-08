"""Loaders for the ranking task."""

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


class RankingwithContextLoader(Loader):
    """Loaders for the ranking tasks, such as argument pair identification."""

    @classmethod
    def default_dataset_file_loaders(cls) -> dict[FileType, FileLoader]:
        """See Loader.default_dataset_file_type."""
        target_field_names = ["context", "query", "true_label"]
        return {
            FileType.datalab: DatalabFileLoader(
                [
                    FileLoaderField("context_column", target_field_names[0], str),
                    FileLoaderField("utterance_column", target_field_names[1], str),
                    FileLoaderField("label_column", target_field_names[2], str),
                ]
            ),
        }

    @classmethod
    def default_output_file_loaders(cls) -> dict[FileType, FileLoader]:
        """See Loader.default_output_file_type."""
        field_name = "predicted_label"
        return {
            FileType.text: TextFileLoader(field_name, str),
            FileType.json: JSONFileLoader(
                [FileLoaderField(field_name, field_name, str)]
            ),
        }
