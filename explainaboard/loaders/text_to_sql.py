from __future__ import annotations

from explainaboard import TaskType
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
from explainaboard.loaders.loader_registry import register_loader


@register_loader(TaskType.text_to_sql)
class TextToSQLLoader(Loader):
    """
    Validate and Reformat system output file with json format:

    """

    @classmethod
    def default_dataset_file_type(cls) -> FileType:
        return FileType.json

    @classmethod
    def default_dataset_file_loaders(cls) -> dict[FileType, FileLoader]:
        target_field_names = ["question", "true_sql", "db_id"]
        return {
            FileType.json: JSONFileLoader(
                [
                    FileLoaderField("question", target_field_names[0], str),
                    FileLoaderField("query", target_field_names[1], str),
                    FileLoaderField("db_id", target_field_names[2], str),
                ]
            ),
        }

    @classmethod
    def default_output_file_loaders(cls) -> dict[FileType, FileLoader]:
        field_name = "predicted_sql"
        target_field_names = ["predicted_sql"]
        return {
            FileType.text: TextFileLoader(field_name, str),
            FileType.tsv: TSVFileLoader(
                [
                    FileLoaderField(0, target_field_names[0], str),
                ],
            ),
        }
