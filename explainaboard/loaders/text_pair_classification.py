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


@register_loader(TaskType.text_pair_classification)
class TextPairClassificationLoader(Loader):
    """
    Validate and Reformat system output file with tsv format:
    text \t true_label \t predicted_label

    usage:
        please refer to `test_loaders.py`
    """

    _default_file_type = FileType.tsv
    _target_names = ["text1", "text2", "true_label", "predicted_label"]
    _default_file_loaders = {
        FileType.tsv: TSVFileLoader(
            [
                FileLoaderField(0, _target_names[0], FileLoaderDType.str, True),
                FileLoaderField(1, _target_names[1], FileLoaderDType.str, True),
                FileLoaderField(2, _target_names[2], FileLoaderDType.str, True),
                FileLoaderField(3, _target_names[3], FileLoaderDType.str, True),
            ],
        ),
        FileType.json: JSONFileLoader(
            [
                FileLoaderField("text1", _target_names[0], FileLoaderDType.str, True),
                FileLoaderField("text2", _target_names[1], FileLoaderDType.str, True),
                FileLoaderField(
                    "true_label", _target_names[2], FileLoaderDType.str, True
                ),
                FileLoaderField(
                    "predicted_label", _target_names[3], FileLoaderDType.str, True
                ),
            ]
        ),
    }
