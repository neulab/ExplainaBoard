"""Loaders for the tabular regression task."""

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


class TabularRegressionLoader(Loader):
    """Loader for the tabular regression task.

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
        target_field_names = ["true_value"]
        return {
            FileType.json: JSONFileLoader(
                [
                    FileLoaderField("true_value", target_field_names[0], float),
                ]
            ),
            FileType.datalab: DatalabFileLoader(
                [
                    FileLoaderField("value_column", target_field_names[0], float),
                ]
            ),
        }

    @classmethod
    def default_output_file_loaders(cls) -> dict[FileType, FileLoader]:
        """See Loader.default_output_file_loaders."""
        field_name = "predicted_value"
        return {
            FileType.text: TextFileLoader(field_name, float),
            FileType.json: JSONFileLoader(
                [FileLoaderField(field_name, field_name, float)]
            ),
        }
