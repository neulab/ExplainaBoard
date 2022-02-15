from typing import Dict, Iterable, List
from explainaboard.constants import *
from .loader import register_loader
from .loader import Loader
from explainaboard.constants import FileType, Source
from explainaboard.tasks import TaskType

@register_loader(TaskType.named_entity_recognition)
class NERLoader(Loader):
    """
    Validate and Reformat system output file with tsv format:
    token \t true_tag \t predicted_tag

    usage:
        please refer to `test_loaders.py`
    """

    def __init__(self, source: Source, file_type: Enum, data :str = None):

        if source == None:
            source = Source.local_filesystem
        if file_type == None:
            file_type = FileType.conll

        self._source = source
        self._file_type = file_type
        self._data = data

    def load(self) -> Iterable[Dict]:
        """
        :param path_system_output: the path of system output file with following format:
        token \t true_tag \t predicted_tag
        :return: class object
        """
        raw_data = self._load_raw_data_points()
        data: List[Dict] = []
        guid = 0
        tokens = []
        ner_true_tags = []
        ner_pred_tags = []


        for id, line in enumerate(raw_data):
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if tokens:
                    data.append({
                        "id": str(guid),
                        "tokens":tokens,
                        "true_tags":ner_true_tags,
                        "pred_tags":ner_pred_tags,
                    })
                    guid += 1
                    tokens = []
                    ner_true_tags = []
                    ner_pred_tags = []
            else:
                # splits = line.split("\t")
                splits = line.split("\t") if len(line.split("\t")) == 3 else line.split(" ")
                tokens.append(splits[0].strip())
                ner_true_tags.append(splits[1].strip())
                ner_pred_tags.append(splits[2].strip())

        # last example
        data.append({
            "id": str(guid),
            "tokens": tokens,
            "true_tags": ner_true_tags,
            "pred_tags": ner_pred_tags
        })
        return data
