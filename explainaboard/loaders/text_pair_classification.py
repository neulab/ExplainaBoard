from explainaboard.constants import FileType
from explainaboard.loaders.file_loader import (
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

    _default_file_type = FileType.tsv
    _target_names = ["text1", "text2", "true_label", "predicted_label"]
    _default_file_loaders = {
        FileType.tsv: TSVFileLoader(
            [
                FileLoaderField(0, _target_names[0], str),
                FileLoaderField(1, _target_names[1], str),
                FileLoaderField(2, _target_names[2], str),
                FileLoaderField(3, _target_names[3], str),
            ],
        ),
        FileType.json: JSONFileLoader(
            [
                FileLoaderField("text1", _target_names[0], str),
                FileLoaderField("text2", _target_names[1], str),
                FileLoaderField("true_label", _target_names[2], str),
                FileLoaderField("predicted_label", _target_names[3], str),
            ]
        ),
    }
