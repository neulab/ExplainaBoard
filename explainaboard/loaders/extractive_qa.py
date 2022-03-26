from explainaboard.constants import FileType
from explainaboard.loaders.file_loader import (
    DatalabFileLoader,
    FileLoaderField,
    JSONFileLoader,
)
from explainaboard.loaders.loader import Loader, register_loader
from explainaboard.tasks import TaskType


@register_loader(TaskType.question_answering_extractive)
class QAExtractiveLoader(Loader):
    _default_file_type = FileType.json
    _target_field_names = ["context", "question", "answers", "predicted_answers"]
    _default_file_loaders = {
        FileType.json: JSONFileLoader(
            [
                FileLoaderField(
                    _target_field_names[0],
                    _target_field_names[0],
                    str,
                    strip_before_parsing=False,
                ),
                FileLoaderField(
                    _target_field_names[1],
                    _target_field_names[1],
                    str,
                    strip_before_parsing=False,
                ),
                FileLoaderField(_target_field_names[2], _target_field_names[2]),
                FileLoaderField(_target_field_names[3], _target_field_names[3]),
            ]
        ),
        FileType.datalab: DatalabFileLoader(
            [
                FileLoaderField(
                    "context",
                    _target_field_names[0],
                    str,
                    strip_before_parsing=False,
                ),
                FileLoaderField(
                    "question",
                    _target_field_names[1],
                    str,
                    strip_before_parsing=False,
                ),
                FileLoaderField("answers", _target_field_names[2]),
                FileLoaderField(
                    "prediction", _target_field_names[3], parser=lambda x: {"text": x}
                ),
            ]
        ),
    }
