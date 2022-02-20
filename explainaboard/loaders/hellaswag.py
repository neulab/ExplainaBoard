from typing import Dict, Iterable, List
from explainaboard.constants import Source, FileType
from enum import Enum
from .loader import register_loader
from .loader import Loader
from datasets import load_dataset
from explainaboard.tasks import TaskType


@register_loader(TaskType.hellaswag)
class HellaswagLoader(Loader):
    """ """

    def __init__(self, source: Source, file_type: Enum, data: str = None):

        if source is None:
            source = Source.local_filesystem
        if file_type is None:
            file_type = FileType.tsv  # easy to make mistake

        self._source = source
        self._file_type = file_type
        self._data = data

    def load(self) -> Iterable[Dict]:
        """
        :param path_system_output: the path of system output file with following format:
        text \t label \t predicted_label
        :return: class object
        """
        dataset = load_dataset('hellaswag')['validation']

        raw_data = self._load_raw_data_points()
        data: List[Dict] = []
        if self._file_type == FileType.tsv:
            for id, dp in enumerate(raw_data):
                sample_id, predicted_label = dp[:2]
                data.append(
                    {
                        "id": str(sample_id),
                        "ind": dataset[int(sample_id)]['ind'],
                        "activity_label": dataset[int(sample_id)]['activity_label'],
                        "ctx_a": dataset[int(sample_id)]['ctx_a'],
                        "ctx_b": dataset[int(sample_id)]['ctx_b'],
                        "ctx": dataset[int(sample_id)]['ctx'],
                        "endings": dataset[int(sample_id)]['endings'],
                        "source_id": dataset[int(sample_id)]['source_id'],
                        "true_label": dataset[int(sample_id)]['label'],
                        "predicted_label": predicted_label.strip(),
                    }
                )
        else:
            raise NotImplementedError
        return data
