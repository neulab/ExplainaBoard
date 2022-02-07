from typing import Dict, Iterable, List
from explainaboard.constants import *
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

    def __init__(self, source: Source, file_type: Enum, data :str = None):

        if source == None:
            source = Source.local_filesystem
        if file_type == None:
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
        raw_data = self._load_raw_data_points() # for json files: loads the entire json
        data: List[Dict] = []
        if self._file_type == FileType.json:
            for id, (link, predictions) in enumerate(raw_data.items()):
                data.append({
                    "id": id,
                    "link": link.strip(),
                    "true_head": link.split('\t')[0].strip(),
                    "true_tail": link.split('\t')[-1].strip(),
                    "predicted_tails": predictions
                })
        else:
            raise NotImplementedError
        return data
