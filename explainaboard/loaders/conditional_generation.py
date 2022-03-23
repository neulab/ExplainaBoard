from explainaboard.constants import FileType
from explainaboard.loaders.file_loader import (
    FileLoaderDType,
    FileLoaderField,
    JSONFileLoader,
    TSVFileLoader,
)
from .loader import register_loader
from .loader import Loader
from explainaboard.tasks import TaskType


@register_loader(TaskType.conditional_generation)
@register_loader(TaskType.summarization)
@register_loader(TaskType.machine_translation)
class ConditionalGenerationLoader(Loader):
    """
    Validate and Reformat system output file with tsv format:
    text \t true_label \t predicted_label

    usage:
        please refer to `test_loaders.py`
    """

    _default_file_type = FileType.tsv
    _field_names = ["source", "reference", "hypothesis"]
    _default_file_loaders = {
        FileType.tsv: TSVFileLoader(
            [
                FileLoaderField(0, _field_names[0], FileLoaderDType.str, True),
                FileLoaderField(1, _field_names[1], FileLoaderDType.str, True),
                FileLoaderField(2, _field_names[2], FileLoaderDType.str, True),
            ],
        ),
        FileType.json: JSONFileLoader(
            [
                FileLoaderField("source", _field_names[0], FileLoaderDType.str, True),
                FileLoaderField(
                    "references", _field_names[1], FileLoaderDType.str, True
                ),
                FileLoaderField(
                    "hypothesis", _field_names[2], FileLoaderDType.str, True
                ),
            ]
        ),
    }
