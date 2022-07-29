from __future__ import annotations

from explainaboard import TaskType
from explainaboard.constants import FileType
from explainaboard.loaders.file_loader import (
    FileLoader,
    FileLoaderField,
    TextFileLoader,
    TSVFileLoader,
)
from explainaboard.loaders.loader import Loader
from explainaboard.loaders.loader_registry import register_loader


@register_loader(TaskType.nlg_meta_evaluation)
class NLGMetaEvaluationLoader(Loader):
    """
    Validate and Reformat system output file with tsv format:
    text \t true_label \t predicted_label

    usage:
        please refer to `test_loaders.py`
    """

    @classmethod
    def default_dataset_file_type(cls) -> FileType:
        return FileType.tsv

    @classmethod
    def default_dataset_file_loaders(cls) -> dict[FileType, FileLoader]:
        target_field_names = [
            'sys_name',
            'seg_id',
            'test_set',
            'src',
            'ref',
            'sys',
            'manual_raw',
            'manual_z',
        ]
        return {
            FileType.tsv: TSVFileLoader(
                [
                    FileLoaderField(0, target_field_names[0], str),
                    FileLoaderField(1, target_field_names[1], str),
                    FileLoaderField(2, target_field_names[2], str),
                    FileLoaderField(3, target_field_names[3], str),
                    FileLoaderField(4, target_field_names[4], str),
                    FileLoaderField(5, target_field_names[5], str),
                    FileLoaderField(6, target_field_names[6], str),
                    FileLoaderField(7, target_field_names[7], str),
                ],
            ),
        }

    @classmethod
    def default_output_file_loaders(cls) -> dict[FileType, FileLoader]:

        field_name = "auto_score"
        return {
            FileType.text: TextFileLoader(field_name, str),
        }
