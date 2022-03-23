from explainaboard.constants import FileType
from explainaboard.loaders.file_loader import (
    FileLoaderDType,
    FileLoaderField,
    JSONFileLoader,
    TSVFileLoader,
)
from explainaboard.tasks import TaskType
from .loader import register_loader
from .loader import Loader


@register_loader(TaskType.aspect_based_sentiment_classification)
class AspectBasedSentimentClassificationLoader(Loader):
    """
    Validate and Reformat system output file with tsv format:
    aspect \t text \t true_label \t predicted_label

    usage:
        please refer to `test_loaders.py`
    """

    _default_file_type = FileType.tsv
    field_names = ["aspect", "text", "true_label", "predicted_label"]
    _default_file_loaders = {
        FileType.tsv: TSVFileLoader(
            [
                FileLoaderField(0, field_names[0], FileLoaderDType.str, True),
                FileLoaderField(1, field_names[1], FileLoaderDType.str, True),
                FileLoaderField(2, field_names[2], FileLoaderDType.str, True),
                FileLoaderField(3, field_names[3], FileLoaderDType.str, True),
            ],
        ),
        FileType.json: JSONFileLoader(
            [
                FileLoaderField(
                    field_names[0], field_names[0], FileLoaderDType.str, True
                ),
                FileLoaderField(
                    field_names[1], field_names[1], FileLoaderDType.str, True
                ),
                FileLoaderField(
                    field_names[2], field_names[2], FileLoaderDType.str, True
                ),
                FileLoaderField(
                    field_names[3], field_names[3], FileLoaderDType.str, True
                ),
            ]
        ),
    }
