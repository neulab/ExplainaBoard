from typing import Iterable, List

from explainaboard.constants import FileType
from explainaboard.loaders.file_loader import JSONFileLoader
from explainaboard.loaders.loader import Loader, register_loader
from explainaboard.tasks import TaskType


@register_loader(TaskType.qa_multiple_choice)
class QAMultipleChoiceLoader(Loader):
    """
    Validate and Reformat system output file with json format:
    "head \t relation \t trueTail": [predTail1, predTail2, predTail3, predTail4, predTail5],

    usage:
        please refer to `test_loaders.py`
    """

    _default_file_type = FileType.json
    _default_file_loaders = {
        FileType.json: JSONFileLoader(None, False),
    }

    def load(self) -> Iterable[dict]:
        """
        :param path_system_output: the path of system output file with following format:
        "head \t relation \t trueTail": [predTail1, predTail2, predTail3, predTail4, predTail5],

        :return: class object
        """
        data: List[dict] = []
        raw_data = self._default_file_loaders[self._file_type].load_raw(
            self._data, self._source
        )

        for id, data_info in enumerate(raw_data):
            data_base = {
                "id": str(id),  # should be string type
                "context": data_info["context"],
                "question": data_info["question"],
                "answers": data_info["answers"],
                "predicted_answers": data_info["predicted_answers"],
            }
            if self.user_defined_features_configs:  # user defined features are present
                # additional user-defined features
                data_base.update(
                    {
                        feature_name: data_info[feature_name]
                        for feature_name in self.user_defined_features_configs
                    }
                )
            data.append(data_base)
        return data
