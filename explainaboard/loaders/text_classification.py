from explainaboard.constants import FileType
from explainaboard.loaders.file_loader import (
    DatalabFileLoader,
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

    _default_file_type = FileType.tsv
    _target_field_names = ["text", "true_label", "predicted_label"]
    _default_file_loaders = {
        FileType.tsv: TSVFileLoader(
            [
                FileLoaderField(0, _target_field_names[0], str),
                FileLoaderField(1, _target_field_names[1], str),
                FileLoaderField(2, _target_field_names[2], str),
            ],
        ),
        FileType.json: JSONFileLoader(
            [
                FileLoaderField("text", _target_field_names[0], str),
                FileLoaderField("true_label", _target_field_names[1], str),
                FileLoaderField("predicted_label", _target_field_names[2], str),
            ]
        ),
        FileType.datalab: DatalabFileLoader(
            [
                FileLoaderField("text", _target_field_names[0], str),
                FileLoaderField("label", _target_field_names[1], str),
                FileLoaderField("prediction", _target_field_names[2], str),
            ]
        ),
    }
