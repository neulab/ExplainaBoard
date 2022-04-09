from __future__ import annotations

from explainaboard.constants import FileType
from explainaboard.loaders.file_loader import (
    FileLoader,
    FileLoaderField,
    JSONFileLoader,
)
from explainaboard.loaders.loader import Loader, register_loader
from explainaboard.tasks import TaskType


@register_loader(TaskType.qa_multiple_choice)
class QAMultipleChoiceLoader(Loader):
    @classmethod
    def default_file_type(cls) -> FileType:
        return FileType.json

    @classmethod
    def default_dataset_file_loaders(cls) -> dict[FileType, FileLoader]:
        target_field_names = ["context", "question", "answers", "predicted_answers"]
        return {
            FileType.json: JSONFileLoader(
                [
                    FileLoaderField("context", target_field_names[0], str),
                    FileLoaderField("question", target_field_names[1], str),
                    FileLoaderField("answers", target_field_names[2], dict),
                    FileLoaderField("predicted_answers", target_field_names[3], dict),
                ]
            ),
        }
