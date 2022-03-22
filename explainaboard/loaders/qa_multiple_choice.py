from typing import Dict, Iterable, List
from explainaboard.constants import FileType
from explainaboard.tasks import TaskType
from .loader import register_loader
from .loader import Loader


@register_loader(TaskType.qa_multiple_choice)
class QAMultipleChoiceLoader(Loader):
    """
    Validate and Reformat system output file with json format:
    "head \t relation \t trueTail": [predTail1, predTail2, predTail3, predTail4, predTail5],

    usage:
        please refer to `test_loaders.py`
    """

    _default_file_type = FileType.json

    def load(self) -> Iterable[Dict]:
        """
        :param path_system_output: the path of system output file with following format:
        "head \t relation \t trueTail": [predTail1, predTail2, predTail3, predTail4, predTail5],

        :return: class object
        """
        super().load()
        data: List[Dict] = []
        if self._file_type == FileType.json:
            for id, data_info in enumerate(self._raw_data):
                data_base = {
                    "id": str(id),  # should be string type
                    "context": data_info["context"],
                    "question": data_info["question"],
                    "answers": data_info["answers"],
                    "predicted_answers": data_info["predicted_answers"],
                }
                if (
                    self.user_defined_features_configs
                ):  # user defined features are present
                    # additional user-defined features
                    data_base.update(
                        {
                            feature_name: data_info[feature_name]
                            for feature_name in self.user_defined_features_configs
                        }
                    )
                data.append(data_base)
        else:
            raise NotImplementedError
        return data
