from explainaboard.constants import FileType
from explainaboard.loaders.file_loader import (
    FileLoader,
    FileLoaderField,
    JSONFileLoader,
    TSVFileLoader,
)
from explainaboard.loaders.loader import Loader, register_loader
from explainaboard.tasks import TaskType


@register_loader(TaskType.aspect_based_sentiment_classification)
class AspectBasedSentimentClassificationLoader(Loader):
    """
    Validate and Reformat system output file with tsv format:
    aspect \t text \t true_label \t predicted_label

    usage:
        please refer to `test_loaders.py`
    """

    @classmethod
    def default_file_type(cls) -> FileType:
        return FileType.tsv

    @classmethod
    def default_dataset_file_loaders(cls) -> dict[FileType, FileLoader]:
        field_names = ["aspect", "text", "true_label", "predicted_label"]
        return {
            FileType.tsv: TSVFileLoader(
                [
                    FileLoaderField(0, field_names[0], str),
                    FileLoaderField(1, field_names[1], str),
                    FileLoaderField(2, field_names[2], str),
                    FileLoaderField(3, field_names[3], str),
                ],
            ),
            FileType.json: JSONFileLoader(
                [
                    FileLoaderField(field_names[0], field_names[0], str),
                    FileLoaderField(field_names[1], field_names[1], str),
                    FileLoaderField(field_names[2], field_names[2], str),
                    FileLoaderField(field_names[3], field_names[3], str),
                ]
            ),
        }
