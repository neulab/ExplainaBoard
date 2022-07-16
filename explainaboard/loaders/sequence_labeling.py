from __future__ import annotations

from explainaboard import TaskType
from explainaboard.constants import FileType
from explainaboard.loaders.file_loader import (
    CoNLLFileLoader,
    DatalabFileLoader,
    FileLoader,
    FileLoaderField,
    JSONFileLoader,
)
from explainaboard.loaders.loader import Loader
from explainaboard.loaders.loader_registry import register_loader


@register_loader(TaskType.chunking)
@register_loader(TaskType.word_segmentation)
@register_loader(TaskType.named_entity_recognition)
class SeqLabLoader(Loader):
    """
    Validate and Reformat system output file with tsv format:
    token \t true_tag \t predicted_tag

    usage:
        please refer to `test_loaders.py`
    """

    @classmethod
    def default_dataset_file_type(cls) -> FileType:
        return FileType.conll

    @classmethod
    def default_output_file_type(cls) -> FileType:
        return FileType.conll

    @classmethod
    def default_dataset_file_loaders(cls) -> dict[FileType, FileLoader]:
        field_names = ["tokens", "true_tags"]
        return {
            FileType.conll: CoNLLFileLoader(
                [
                    FileLoaderField(0, field_names[0], str),
                    FileLoaderField(1, field_names[1], str),
                ]
            ),
            FileType.datalab: DatalabFileLoader(
                [
                    FileLoaderField("tokens_column", field_names[0], list),
                    FileLoaderField("tags_column", field_names[1], list),
                ]
            ),
        }

    @classmethod
    def default_output_file_loaders(cls) -> dict[FileType, FileLoader]:
        return {
            FileType.conll: CoNLLFileLoader([FileLoaderField(1, "pred_tags", str)]),
            FileType.json: JSONFileLoader(
                [
                    FileLoaderField("tokens", "tokens", list),
                    FileLoaderField("predicted_tags", "pred_tags", list),
                ]
            ),
        }
