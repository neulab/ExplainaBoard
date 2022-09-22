"""Loaders for the aspect based sentiment classification task."""

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


@register_loader(TaskType.aspect_based_sentiment_classification)
class AspectBasedSentimentClassificationLoader(Loader):
    """Loader for the aspect based sentiment classification task.

    usage:
        please refer to `test_loaders.py`
    """

    @classmethod
    def default_dataset_file_type(cls) -> FileType:
        """See Loader.default_dataset_file_type."""
        return FileType.tsv

    @classmethod
    def default_dataset_file_loaders(cls) -> dict[FileType, FileLoader]:
        """See Loader.default_dataset_file_loaders."""
        field_names = ["aspect", "text", "true_label"]
        return {
            FileType.tsv: TSVFileLoader(
                [
                    FileLoaderField(0, field_names[0], str),
                    FileLoaderField(1, field_names[1], str),
                    FileLoaderField(2, field_names[2], str),
                ],
            ),
            FileType.json: JSONFileLoader(
                [
                    FileLoaderField(field_names[0], field_names[0], str),
                    FileLoaderField(field_names[1], field_names[1], str),
                    FileLoaderField(field_names[2], field_names[2], str),
                ]
            ),
            FileType.datalab: DatalabFileLoader(
                [
                    FileLoaderField("span_column", field_names[0], str),
                    FileLoaderField("text", field_names[1], str),
                    FileLoaderField("label", field_names[2], str),
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
                [FileLoaderField(field_name, field_name, str)]
            ),
        }
