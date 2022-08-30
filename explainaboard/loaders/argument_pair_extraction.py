from __future__ import annotations

from explainaboard import TaskType
from explainaboard.constants import FileType
from explainaboard.loaders.file_loader import (
    CoNLLFileLoader,
    DatalabFileLoader,
    FileLoader,
    FileLoaderField,
)
from explainaboard.loaders.loader import Loader
from explainaboard.loaders.loader_registry import register_loader


@register_loader(TaskType.argument_pair_extraction)
class ArgumentPairExtraction(Loader):
    """
    Validate and Reformat system output file with tsv format:
    sentence \t true_tag \t predicted_tag

    usage:
        please refer to `test_loaders.py`
    """

    @classmethod
    def default_dataset_file_type(cls) -> FileType:
        return FileType.datalab

    @classmethod
    def default_output_file_type(cls) -> FileType:
        return FileType.conll

    @classmethod
    def default_dataset_file_loaders(cls) -> dict[FileType, FileLoader]:
        field_names = ["sentences", "true_tags"]
        return {
            FileType.datalab: DatalabFileLoader(
                [
                    FileLoaderField("sentences_column", field_names[0], list),
                    FileLoaderField("labels_column", field_names[1], list),
                ]
            ),
        }

    @classmethod
    def default_output_file_loaders(cls) -> dict[FileType, FileLoader]:
        return {
            FileType.conll: CoNLLFileLoader([FileLoaderField(1, "pred_tags", str)]),
        }
