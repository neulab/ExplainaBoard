from explainaboard.constants import FileType
from explainaboard.loaders.file_loader import (
    FileLoaderField,
    JSONFileLoader,
    TSVFileLoader,
)
from explainaboard.loaders.loader import Loader, register_loader
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
                FileLoaderField(0, _field_names[0], str),
                FileLoaderField(1, _field_names[1], str),
                FileLoaderField(2, _field_names[2], str),
            ],
        ),
        FileType.json: JSONFileLoader(
            [
                FileLoaderField("source", _field_names[0], str),
                FileLoaderField("references", _field_names[1], str),
                FileLoaderField("hypothesis", _field_names[2], str),
            ]
        ),
    }
