"""Loaders for the text classification task."""

from __future__ import annotations

from explainaboard.constants import FileType
from explainaboard.loaders.file_loader import (
    DatalabFileLoader,
    FileLoader,
    FileLoaderField,
    JSONFileLoader,
    TextFileLoader,
    TSVFileLoader,
)
from explainaboard.loaders.loader import Loader


class TextClassificationLoader(Loader):
    """Loader for the text classification task.

    usage:
        please refer to `loaders_test.py`
    """

    @classmethod
    def default_dataset_file_type(cls) -> FileType:
        """See Loader.default_dataset_file_type."""
        return FileType.tsv

    @classmethod
    def default_dataset_file_loaders(cls) -> dict[FileType, FileLoader]:
        """See Loader.default_dataset_file_loaders."""
        target_field_names = ["text", "true_label"]
        return {
            FileType.tsv: TSVFileLoader(
                [
                    FileLoaderField(0, target_field_names[0], str),
                    FileLoaderField(1, target_field_names[1], str),
                ],
            ),
            FileType.json: JSONFileLoader(
                [
                    FileLoaderField("text", target_field_names[0], str),
                    FileLoaderField("true_label", target_field_names[1], str),
                ]
            ),
            FileType.datalab: DatalabFileLoader(
                [
                    FileLoaderField("text_column", target_field_names[0], str),
                    FileLoaderField("label_column", target_field_names[1], str),
                ]
            ),
        }

    @classmethod
    def default_output_file_loaders(cls) -> dict[FileType, FileLoader]:
        """See Loader.default_output_file_loaders."""
        field_name = "predicted_label"
        return {
            FileType.text: TextFileLoader(field_name, str),
            FileType.json: JSONFileLoader(
                [
                    FileLoaderField(field_name, field_name, str),
                    FileLoaderField(
                        "confidence", "confidence", dtype=float, optional=True
                    ),
                ]
            ),
        }
