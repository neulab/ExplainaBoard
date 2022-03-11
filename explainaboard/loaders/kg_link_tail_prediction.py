from typing import Dict, Iterable, List
from explainaboard.constants import Source, FileType
from enum import Enum
from explainaboard.tasks import TaskType
from .loader import register_loader
from .loader import Loader


@register_loader(TaskType.kg_link_tail_prediction)
class KgLinkTailPredictionLoader(Loader):
    """
    Validate and Reformat system output file with json format:
    "head \t relation \t trueTail": [predTail1, predTail2, predTail3, predTail4, predTail5],

    usage:
        please refer to `test_loaders.py`
    """

    def __init__(self, source: Source, file_type: Enum, data: str = None):

        if source is None:
            source = Source.local_filesystem
        if file_type is None:
            file_type = FileType.json

        self._source = source
        self._file_type = file_type
        self._data = data
        self.user_defined_features_configs = None

    def load_user_defined_features_configs(self):

        raw_data = self._load_raw_data_points()  # for json files: loads the entire json
        self.user_defined_features_configs = raw_data.get(
            "user_defined_features_configs", None
        )
        return self.user_defined_features_configs

    def load(self) -> Iterable[Dict]:
        """
        :param path_system_output: the path of system output file with following format:
        "head \t relation \t trueTail": [predTail1, predTail2, predTail3, predTail4, predTail5],

        :return: class object
        """
        raw_data = self._load_raw_data_points()  # for json files: loads the entire json
        data: List[Dict] = []
        if self._file_type == FileType.json:
            if (
                self.user_defined_features_configs is not None
            ):  # user defined features are present
                for id, (link, features_dict) in enumerate(
                    raw_data['predictions'].items()
                ):

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
                            for feature_name in self.user_defined_features_configs.keys()
                        }
                    )

                    # save
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
        else:
            raise NotImplementedError
        return data
