from explainaboard.constants import FileType
from explainaboard.loaders.file_loader import (
    DatalabFileLoader,
    FileLoaderDType,
    FileLoaderField,
    JSONFileLoader,
)
from .loader import register_loader
from .loader import Loader
from explainaboard.tasks import TaskType


@register_loader(TaskType.question_answering_extractive)
class QAExtractiveLoader(Loader):
    _default_file_type = FileType.json
    _target_field_names = ["context", "question", "answers", "predicted_answers"]
    _default_file_loaders = {
        FileType.json: JSONFileLoader(
            [
                FileLoaderField(
                    _target_field_names[0], _target_field_names[0], FileLoaderDType.str
                ),
                FileLoaderField(
                    _target_field_names[1], _target_field_names[1], FileLoaderDType.str
                ),
                FileLoaderField(
                    _target_field_names[2],
                    _target_field_names[2],
                    FileLoaderDType.other,
                ),
                FileLoaderField(
                    _target_field_names[3],
                    _target_field_names[3],
                    FileLoaderDType.other,
                ),
            ]
        ),
        FileType.datalab: DatalabFileLoader(
            [
                FileLoaderField("context", _target_field_names[0], FileLoaderDType.str),
                FileLoaderField(
                    "question", _target_field_names[1], FileLoaderDType.str
                ),
                FileLoaderField(
                    "answers", _target_field_names[2], FileLoaderDType.other
                ),
                FileLoaderField(
                    "prediction",
                    _target_field_names[3],
                    FileLoaderDType.other,
                    parser=lambda x: {"text": x},
                ),
            ]
        ),
    }
