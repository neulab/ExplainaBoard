from explainaboard.constants import FileType
from explainaboard.loaders.file_loader import (
    FileLoader,
    FileLoaderField,
    JSONFileLoader,
    TSVFileLoader,
)
from explainaboard.loaders.loader import Loader, register_loader
from explainaboard.tasks import TaskType


@register_loader(TaskType.text_pair_classification)
class TextPairClassificationLoader(Loader):
    """
    Validate and Reformat system output file with tsv format:
    text \t true_label \t predicted_label

    usage:
        please refer to `test_loaders.py`
    """

    @classmethod
    def default_file_type(cls) -> FileType:
        return FileType.tsv

    @classmethod
    def default_dataset_file_loaders(cls) -> dict[FileType, FileLoader]:
        target_names = ["text1", "text2", "true_label", "predicted_label"]
        return {
            FileType.tsv: TSVFileLoader(
                [
                    FileLoaderField(0, target_names[0], str),
                    FileLoaderField(1, target_names[1], str),
                    FileLoaderField(2, target_names[2], str),
                    FileLoaderField(3, target_names[3], str),
                ],
            ),
            FileType.json: JSONFileLoader(
                [
                    FileLoaderField("text1", target_names[0], str),
                    FileLoaderField("text2", target_names[1], str),
                    FileLoaderField("true_label", target_names[2], str),
                    FileLoaderField("predicted_label", target_names[3], str),
                ]
            ),
        }
