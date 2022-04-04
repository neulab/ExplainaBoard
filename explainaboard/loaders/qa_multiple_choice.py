from __future__ import annotations

from explainaboard.constants import FileType
from explainaboard.loaders.file_loader import FileLoaderField, JSONFileLoader
from explainaboard.loaders.loader import Loader, register_loader
from explainaboard.tasks import TaskType


@register_loader(TaskType.qa_multiple_choice)
class QAMultipleChoiceLoader(Loader):

    _default_file_type = FileType.json
    _target_field_names = ["context", "question", "answers", "predicted_answers"]
    _default_file_loaders = {
        FileType.json: JSONFileLoader(
            [
                FileLoaderField("context", _target_field_names[0], str),
                FileLoaderField("question", _target_field_names[1], str),
                # TODO(Pengfei): I add the `dict` type for temporal use, but
                #  wonder if we need to generalize the current type mechanism
                FileLoaderField("answers", _target_field_names[2], dict),
                FileLoaderField("predicted_answers", _target_field_names[3], dict),
            ]
        ),
    }
