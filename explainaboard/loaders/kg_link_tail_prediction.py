from __future__ import annotations

import json

from explainaboard import TaskType
from explainaboard.constants import FileType
from explainaboard.loaders.file_loader import FileLoader, JSONFileLoader
from explainaboard.loaders.loader import Loader
from explainaboard.loaders.loader_registry import register_loader
from explainaboard.utils import cache_api


@register_loader(TaskType.kg_link_tail_prediction)
class KgLinkTailPredictionLoader(Loader):
    """
    Validate and Reformat system output file with json format:
    "head \t relation \t trueTail": [predTail1, predTail2, ..., predTail5],

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
        return FileType.json

    @classmethod
    def default_output_file_type(cls) -> FileType:
        return FileType.json

    @classmethod
    def default_dataset_file_loaders(cls) -> dict[FileType, FileLoader]:
        return {FileType.json: JSONFileLoader(None, False)}

    @classmethod
    def default_output_file_loaders(cls) -> dict[FileType, FileLoader]:
        return {FileType.json: JSONFileLoader(None, False)}

    def load(self) -> list[dict]:
        """
        :param path_system_output:
            the path of system output file with following format:
            "head \t relation \t trueTail": [predTail1, predTail2, ..., predTail5],

        :return: class object
        """
        data: list[dict] = []

        # TODO(odashi):
        # Avoid potential bug: load_raw returns Iterable[Any] which is not a dict.
        raw_data: dict[str, dict] = self._output_file_loader.load_raw(  # type: ignore
            self._output_data, self._output_source
        )

        # Map entity into an interpretable version
        entity_dic = {}
        file_path = cache_api.cache_online_file(
            'http://phontron.com/download/explainaboard/pre_computed/kg/entity2wikidata.json',  # noqa
            'pre_computed/kg/entity2wikidata.json',
        )
        with open(file_path, 'r') as file:
            entity_dic = json.loads(file.read())

        if self.user_defined_features_configs:  # user defined features are present
            for example_id, features_dict in raw_data.items():
                data_i = {
                    "id": str(example_id),  # should be string type
                    "true_head": entity_dic[features_dict["gold_head"]]["label"]
                    if features_dict["gold_head"] in entity_dic.keys()
                    else features_dict["gold_head"],
                    "true_link": features_dict["gold_predicate"],
                    "true_tail": entity_dic[features_dict["gold_tail"]]["label"]
                    if features_dict["gold_tail"] in entity_dic.keys()
                    else features_dict["gold_tail"],
                    "true_tail_anonymity": features_dict["gold_tail"],
                    "true_label": features_dict[
                        "gold_" + features_dict["predict"]
                    ],  # the entity to which we compare the predictions
                    "predictions": features_dict["predictions"],
                }

                # additional user-defined features
                data_i.update(
                    {
                        feature_name: features_dict[feature_name]
                        for feature_name in self.user_defined_features_configs
                    }
                )

                data.append(data_i)
        else:
            for _, (example_id, features_dict) in enumerate(raw_data.items()):
                data.append(
                    {
                        "id": str(example_id),  # should be string type
                        "true_head": entity_dic[features_dict["gold_head"]]["label"]
                        if features_dict["gold_head"] in entity_dic.keys()
                        else features_dict["gold_head"],
                        "true_link": features_dict["gold_predicate"],
                        "true_tail": entity_dic[features_dict["gold_tail"]]["label"]
                        if features_dict["gold_tail"] in entity_dic.keys()
                        else features_dict["gold_tail"],
                        "true_tail_anonymity": features_dict["gold_tail"],
                        "true_label": features_dict[
                            "gold_" + features_dict["predict"]
                        ],  # the entity to which we compare the predictions
                        "predictions": features_dict["predictions"],
                    }
                )
        return data
