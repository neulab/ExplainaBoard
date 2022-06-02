from __future__ import annotations

from explainaboard.constants import FileType, TaskType
from explainaboard.loaders.file_loader import (
    DatalabFileLoader,
    FileLoader,
    FileLoaderField,
    JSONFileLoader,
)
from explainaboard.loaders.loader import Loader
from explainaboard.loaders.loader_registry import register_loader


@register_loader(TaskType.grammatical_error_correction)
class GrammaticalErrorCorrectionLoader(Loader):
    """
    Validate and Reformat system output file with tsv format:
    text \t true_label \t predicted_label

    usage:
        please refer to `test_loaders.py`
    """

    JSON_FIELDS: list[str | tuple[str, str]] = ['text', 'edits']

    @classmethod
    def default_dataset_file_type(cls) -> FileType:
        return FileType.json

    @classmethod
    def default_output_file_type(cls) -> FileType:
        return FileType.json

    @classmethod
    def default_dataset_file_loaders(cls) -> dict[FileType, FileLoader]:
        return {
            FileType.json: JSONFileLoader(
                [
                    FileLoaderField('text', 'text', str),
                    FileLoaderField('edits', 'edits', dict),
                ]
            ),
            FileType.datalab: DatalabFileLoader(
                [
                    FileLoaderField('text', 'text', str),
                    FileLoaderField('edits', 'edits', dict),
                ]
            ),
        }

    @classmethod
    def default_output_file_loaders(cls) -> dict[FileType, FileLoader]:
        field_name = "predicted_edits"
        return {
            FileType.json: JSONFileLoader(
                [FileLoaderField(field_name, field_name, dict)]
            ),
        }
