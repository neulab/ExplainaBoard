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
    "head \t relation \t trueTail": [predTail1, predTail2, predTail3, predTail4, predTail5],

    usage:
        please refer to `test_loaders.py`
    """

    _default_file_type = FileType.json
    _default_file_loaders = {FileType.json: JSONFileLoader(None, False)}

    def load(self) -> Iterable[dict]:
        """
        :param path_system_output: the path of system output file with following format:
        "head \t relation \t trueTail": [predTail1, predTail2, predTail3, predTail4, predTail5],

        :return: class object
        """
        data: list[dict] = []

        # TODO(odashi): Avoid potential bug: load_raw returns Iterable[Any] which is not a dict.
        raw_data: dict[str, dict[str, str]] = self.file_loaders[  # type: ignore
            unwrap(self._file_type)
        ].load_raw(self._data, self._source)

        if self.user_defined_features_configs:  # user defined features are present
            for id, (link, features_dict) in enumerate(raw_data.items()):

                data_i = {
                    "id": str(id),  # should be string type
                    "link": link.strip(),
                    "relation": link.split('\t')[1].strip(),
                    "true_head": link.split('\t')[0].strip(),
                    "true_tail": link.split('\t')[-1].strip(),
                    "predicted_tails": features_dict["predictions"],
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
            for id, (link, predictions) in enumerate(raw_data.items()):
                data.append(
                    {
                        "id": str(id),  # should be string type
                        "link": link.strip(),
                        "relation": link.split('\t')[1].strip(),
                        "true_head": link.split('\t')[0].strip(),
                        "true_tail": link.split('\t')[-1].strip(),
                        "predicted_tails": predictions,
                    }
                )
        return data
