from __future__ import annotations

from explainaboard import TaskType
from explainaboard.constants import FileType
from explainaboard.loaders.file_loader import (
    DatalabFileLoader,
    FileLoader,
    FileLoaderField,
    JSONFileLoader,
    TextFileLoader,
)
from explainaboard.loaders.loader import Loader
from explainaboard.loaders.loader_registry import register_loader


@register_loader(TaskType.tabular_regression)
class TabularRegressionLoader(Loader):
    """
    Validate and Reformat system output file with tsv format:
    text \t true_label \t predicted_label

    usage:
        please refer to `test_loaders.py`
    """

    @classmethod
    def default_dataset_file_type(cls) -> FileType:
        return FileType.json

    @classmethod
    def default_dataset_file_loaders(cls) -> dict[FileType, FileLoader]:
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
        field_name = "predicted_value"
        return {
            FileType.text: TextFileLoader(field_name, float),
            FileType.json: JSONFileLoader(
                [FileLoaderField(field_name, field_name, float)]
            ),
        }
