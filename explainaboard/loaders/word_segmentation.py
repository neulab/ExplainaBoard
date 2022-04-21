from __future__ import annotations

from explainaboard import TaskType
from explainaboard.constants import FileType
from explainaboard.loaders.file_loader import (
    CoNLLFileLoader,
    FileLoader,
    FileLoaderField,
)
from explainaboard.loaders.loader import Loader
from explainaboard.loaders.loader_registry import register_loader


@register_loader(TaskType.word_segmentation)
class WSLoader(Loader):
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
        return {
            FileType.conll: CoNLLFileLoader(
                [
                    FileLoaderField(0, "tokens", str),
                    FileLoaderField(1, "true_tags", str),
                ]
            )
        }

    @classmethod
    def default_output_file_loaders(cls) -> dict[FileType, FileLoader]:
        return {FileType.conll: CoNLLFileLoader([FileLoaderField(1, "pred_tags", str)])}
