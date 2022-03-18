from typing import Dict, Iterable, List
from explainaboard.constants import Source, FileType
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

    def __init__(self, source: Source, file_type: FileType, data: str = None):

        if source is None:
            source = Source.local_filesystem
        if file_type is None:
            file_type = FileType.json

        self._source = source
        self._file_type = file_type
        self._data = data

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
                data.append(
                    {
                        "id": str(id),  # should be string type
                        "context": data_info["context"],
                        "question": data_info["question"],
                        "answers": data_info["answers"],
                        "predicted_answers": data_info["predicted_answers"],
                    }
                )
        else:
            raise NotImplementedError
        return data
