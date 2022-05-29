from __future__ import annotations

from explainaboard import TaskType
from explainaboard.constants import FileType
from explainaboard.loaders.file_loader import (
    DatalabFileLoader,
    FileLoader,
    FileLoaderField,
    JSONFileLoader,
)
from explainaboard.loaders.loader import Loader
from explainaboard.loaders.loader_registry import register_loader


@register_loader(TaskType.cloze_mutiple_choice)
class ClozeMultipleChoiceLoader(Loader):
    @classmethod
    def default_dataset_file_type(cls) -> FileType:
        return FileType.json

    @classmethod
    def default_output_file_type(cls) -> FileType:
        return FileType.json

    @classmethod
    def default_dataset_file_loaders(cls) -> dict[FileType, FileLoader]:
        target_field_names = ["context", "options", "question_mark", "answers"]
        return {
            FileType.json: JSONFileLoader(
                [
                    FileLoaderField("context", target_field_names[0], str),
                    FileLoaderField("options", target_field_names[1], list),
                    FileLoaderField("question_mark", target_field_names[2], str),
                    FileLoaderField("answers", target_field_names[3], dict),
                ]
            ),
            FileType.datalab: DatalabFileLoader(
                [
                    FileLoaderField("context", target_field_names[0], str),
                    FileLoaderField("options", target_field_names[1], list),
                    FileLoaderField("question_mark", target_field_names[2], str),
                    FileLoaderField("answers", target_field_names[3], dict),
                ]
            ),
        }

    @classmethod
    def default_output_file_loaders(cls) -> dict[FileType, FileLoader]:
        return {
            FileType.json: JSONFileLoader(
                [FileLoaderField("predicted_answers", "predicted_answers", dict)]
            )
        }
