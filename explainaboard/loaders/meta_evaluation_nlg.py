"""Loaders for the NLG meta evaluation task."""

from __future__ import annotations

from explainaboard.constants import FileType
from explainaboard.loaders.file_loader import (
    DatalabFileLoader,
    FileLoader,
    FileLoaderField,
    JSONFileLoader,
)
from explainaboard.loaders.loader import Loader


class MetaEvaluationNLGLoader(Loader):
    """Loader for the natural language generation task."""

    @classmethod
    def default_dataset_file_type(cls) -> FileType:
        """See Loader.default_dataset_file_type."""
        return FileType.datalab

    @classmethod
    def default_dataset_file_loaders(cls) -> dict[FileType, FileLoader]:
        """See Loader.default_dataset_file_loaders."""
        return {
            FileType.datalab: DatalabFileLoader(
                [
                    FileLoaderField("source_column", "source", str),
                    FileLoaderField("references_column", "references", list),
                    FileLoaderField("hypotheses_column", "hypotheses", dict),
                    FileLoaderField("scores_column", "manual_scores", list),
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
        return {
            FileType.json: JSONFileLoader(
                [
                    FileLoaderField("auto_scores", "auto_scores", list),
                ]
            ),
        }
