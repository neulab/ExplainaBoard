"""Loaders for the NLG meta evaluation task."""

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


@register_loader(TaskType.meta_evaluation_nlg)
class MetaEvaluationNLGLoader(Loader):
    """Loader for the natural language generation task.

    usage:
        please refer to `test_loaders.py`
    """

    @classmethod
    def default_dataset_file_type(cls) -> FileType:
        """See Loader.default_dataset_file_type."""
        return FileType.datalab

    @classmethod
    def default_dataset_file_loaders(cls) -> dict[FileType, FileLoader]:
        """See Loader.default_dataset_file_loaders."""
        target_field_names = ['source', 'references', 'hypotheses', 'manual_scores']
        return {
            FileType.datalab: DatalabFileLoader(
                [
                    FileLoaderField("source_column", target_field_names[0], str),
                    FileLoaderField("references_column", target_field_names[1], list),
                    FileLoaderField("hypotheses_column", target_field_names[2], dict),
                    FileLoaderField("scores_column", target_field_names[3], list),
                ],
            ),
        }

    @classmethod
    def default_output_file_type(cls) -> FileType:
        """See Loader.default_output_file_type."""
        return FileType.json

    @classmethod
    def default_output_file_loaders(cls) -> dict[FileType, FileLoader]:
        """See Loader.default_output_file_loaders."""
        field_name = "auto_scores"
        return {
            FileType.json: JSONFileLoader(
                [
                    FileLoaderField("auto_scores", field_name, list),
                ]
            ),
        }
