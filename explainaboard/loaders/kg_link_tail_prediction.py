"""Loaders for knowledge graph link prediction tasks."""

from __future__ import annotations

import json

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
from explainaboard.utils import cache_api
from explainaboard.utils.preprocessor import KGMapPreprocessor


@register_loader(TaskType.kg_link_tail_prediction)
class KgLinkTailPredictionLoader(Loader):
    """Loader for the knowledge graph link prediction task.

    usage:
        please refer to `test_loaders.py`

    NOTE: kg task has a system output format that's different from all the
    other tasks. Samples are stored in a dict instead of a list so we have
    special loading logic implemented here. We have plans to change this in
    the in the future. Also, the dataset and the output is stored in the same
    file so the dataset file loader doesn't do anything. We also plan to change
    this behavior in the future.
    """

    @classmethod
    def default_dataset_file_type(cls) -> FileType:
        """See Loader.default_dataset_file_type."""
        return FileType.json

    @classmethod
    def default_output_file_type(cls) -> FileType:
        """See Loader.default_output_file_type."""
        return FileType.json

    @classmethod
    def default_dataset_file_loaders(cls) -> dict[FileType, FileLoader]:
        """See Loader.default_dataset_file_loaders."""
        file_path = cache_api.cache_online_file(
            'https://storage.googleapis.com/inspired-public-data/'
            'explainaboard/task_data/kg_link_tail_prediction/entity2wikidata.json',
            'explainaboard/task_data/kg_link_tail_prediction/entity2wikidata.json',
        )
        with open(file_path, 'r') as file:
            entity_dic = json.loads(file.read())

        map_preprocessor = KGMapPreprocessor(resources={"dictionary": entity_dic})

        target_field_names = [
            "true_head",
            "true_head_decipher",
            "true_link",
            "true_tail",
            "true_tail_decipher",
        ]
        return {
            FileType.json: JSONFileLoader(
                [
                    FileLoaderField("gold_head", target_field_names[0], str),
                    FileLoaderField(
                        "gold_head", target_field_names[1], str, parser=map_preprocessor
                    ),
                    FileLoaderField("gold_predicate", target_field_names[2], str),
                    FileLoaderField("gold_tail", target_field_names[3], str),
                    FileLoaderField(
                        "gold_tail", target_field_names[4], str, parser=map_preprocessor
                    ),
                ]
            ),
            FileType.datalab: DatalabFileLoader(
                [
                    FileLoaderField("head_column", target_field_names[0], str),
                    FileLoaderField(
                        "head_column",
                        target_field_names[1],
                        str,
                        parser=map_preprocessor,
                    ),
                    FileLoaderField("link_column", target_field_names[2], str),
                    FileLoaderField("tail_column", target_field_names[3], str),
                    FileLoaderField(
                        "tail_column",
                        target_field_names[4],
                        str,
                        parser=map_preprocessor,
                    ),
                ]
            ),
        }

    @classmethod
    def default_output_file_loaders(cls) -> dict[FileType, FileLoader]:
        """See Loader.default_output_file_loaders."""
        target_field_names = ["predict", "predictions", "true_rank"]
        return {
            FileType.json: JSONFileLoader(
                [
                    FileLoaderField("predict", target_field_names[0], str),
                    FileLoaderField("predictions", target_field_names[1], list),
                    FileLoaderField("true_rank", target_field_names[2], int),
                ]
            )
        }
