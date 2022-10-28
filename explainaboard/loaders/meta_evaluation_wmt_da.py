"""Loaders for the NLG meta evaluation task."""

from __future__ import annotations

from explainaboard.constants import FileType
from explainaboard.loaders.file_loader import (
    DatalabFileLoader,
    FileLoader,
    FileLoaderField,
    TextFileLoader,
    TSVFileLoader,
)
from explainaboard.loaders.loader import Loader


class MetaEvaluationWMTDALoader(Loader):
    """Loader for the natural language generation task.

    usage:
        please refer to `loaders_test.py`
    """

    @classmethod
    def default_dataset_file_type(cls) -> FileType:
        """See Loader.default_dataset_file_type."""
        return FileType.tsv

    @classmethod
    def default_dataset_file_loaders(cls) -> dict[FileType, FileLoader]:
        """See Loader.default_dataset_file_loaders."""
        target_field_names = [
            "sys_name",
            "seg_id",
            "test_set",
            "src",
            "ref",
            "sys",
            "manual_raw",
            "manual_z",
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
                    FileLoaderField(6, target_field_names[6], float),
                    FileLoaderField(7, target_field_names[7], float),
                ],
            ),
            FileType.datalab: DatalabFileLoader(
                [
                    FileLoaderField("sys_name_column", target_field_names[0], str),
                    FileLoaderField("seg_id_column", target_field_names[1], str),
                    FileLoaderField("test_set_column", target_field_names[2], str),
                    FileLoaderField("source_column", target_field_names[3], str),
                    FileLoaderField("reference_column", target_field_names[4], str),
                    FileLoaderField("hypothesis_column", target_field_names[5], str),
                    FileLoaderField(
                        "manual_score_raw_column", target_field_names[6], str
                    ),
                    FileLoaderField(
                        "manual_score_z_column", target_field_names[7], str
                    ),
                ]
            ),
        }

    @classmethod
    def default_output_file_loaders(cls) -> dict[FileType, FileLoader]:
        """See Loader.default_output_file_loaders."""
        field_name = "auto_score"
        return {
            FileType.text: TextFileLoader(field_name, float),
        }
