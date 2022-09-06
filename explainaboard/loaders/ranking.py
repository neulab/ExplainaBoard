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


@register_loader(TaskType.argument_pair_identification)
@register_loader(TaskType.ranking)
class RankingLoader(Loader):
    """
    Validate and Reformat system output file with tsv format:
    text \t true_label \t predicted_label

    usage:
        please refer to `test_loaders.py`
    """

    @classmethod
    def default_dataset_file_loaders(cls) -> dict[FileType, FileLoader]:
        target_field_names = ["context", "utterance", "true_label"]
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
        field_name = "predicted_label"
        return {
            FileType.text: TextFileLoader(field_name, str),
            FileType.json: JSONFileLoader(
                [FileLoaderField(field_name, field_name, str)]
            ),
        }
