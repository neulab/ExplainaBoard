from explainaboard.constants import FileType
from explainaboard.loaders.file_loader import (
    DatalabFileLoader,
    FileLoader,
    FileLoaderField,
    JSONFileLoader,
    TSVFileLoader,
)
from explainaboard.loaders.loader import Loader, register_loader
from explainaboard.tasks import TaskType


@register_loader(TaskType.text_classification)
class TextClassificationLoader(Loader):
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
        target_field_names = ["text", "true_label", "predicted_label"]
        return {
            FileType.tsv: TSVFileLoader(
                [
                    FileLoaderField(0, target_field_names[0], str),
                    FileLoaderField(1, target_field_names[1], str),
                    FileLoaderField(2, target_field_names[2], str),
                ],
            ),
            FileType.json: JSONFileLoader(
                [
                    FileLoaderField("text", target_field_names[0], str),
                    FileLoaderField("true_label", target_field_names[1], str),
                    FileLoaderField("predicted_label", target_field_names[2], str),
                ]
            ),
            FileType.datalab: DatalabFileLoader(
                [
                    FileLoaderField("text", target_field_names[0], str),
                    FileLoaderField("label", target_field_names[1], str),
                    FileLoaderField("prediction", target_field_names[2], str),
                ]
            ),
        }
