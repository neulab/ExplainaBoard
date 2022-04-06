from __future__ import annotations

from collections.abc import Iterable

from explainaboard.constants import FileType
from explainaboard.loaders.file_loader import JSONFileLoader
from explainaboard.loaders.loader import Loader, register_loader
from explainaboard.tasks import TaskType
from explainaboard.utils.typing_utils import unwrap


@register_loader(TaskType.kg_link_tail_prediction)
class KgLinkTailPredictionLoader(Loader):
    """
    Validate and Reformat system output file with json format:
    "head \t relation \t trueTail": [predTail1, predTail2, ..., predTail5],

    usage:
        please refer to `test_loaders.py`
    """

    _default_file_type = FileType.json
    _default_file_loaders = {FileType.json: JSONFileLoader(None, False)}

    def load(self) -> Iterable[dict]:
        """
        :param path_system_output:
            the path of system output file with following format:
            "head \t relation \t trueTail": [predTail1, predTail2, ..., predTail5],

        :return: class object
        """
        data: list[dict] = []

        # TODO(odashi):
        # Avoid potential bug: load_raw returns Iterable[Any] which is not a dict.
        raw_data: dict[str, dict[str, str]] = self.file_loaders[  # type: ignore
            unwrap(self._file_type)
        ].load_raw(self._data, self._source)

        if self.user_defined_features_configs:  # user defined features are present
            for _, (example_id, features_dict) in enumerate(raw_data.items()):

                data_i = {
                    "id": str(example_id),  # should be string type
                    "true_head": features_dict["gold_head"],
                    "true_link": features_dict["gold_predicate"],
                    "true_tail": features_dict["gold_tail"],
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
                        "true_head": features_dict["gold_head"],
                        "true_link": features_dict["gold_predicate"],
                        "true_tail": features_dict["gold_tail"],
                        "true_label": features_dict[
                            "gold_" + features_dict["predict"]
                        ],  # the entity to which we compare the predictions
                        "predictions": features_dict["predictions"],
                    }
                )
        return data
