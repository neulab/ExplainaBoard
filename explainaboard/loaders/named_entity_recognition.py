from typing import Dict, Iterable, List

from .loader import register_loader
from .loader import Loader
from explainaboard.constants import FileType
from explainaboard.tasks import TaskType


@register_loader(TaskType.named_entity_recognition)
class NERLoader(Loader):
    """
    Validate and Reformat system output file with tsv format:
    token \t true_tag \t predicted_tag

    usage:
        please refer to `test_loaders.py`
    """

    _default_file_type = FileType.conll

    def load(self) -> Iterable[Dict]:
        """
        :param path_system_output: the path of system output file with following format:
        token \t true_tag \t predicted_tag
        :return: class object
        """
        super().load()
        data: List[Dict] = []
        guid = 0
        tokens = []
        ner_true_tags = []
        ner_pred_tags = []

        for id, line in enumerate(self._raw_data):
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if tokens:
                    data.append(
                        {
                            "id": str(guid),
                            "tokens": tokens,
                            "true_tags": ner_true_tags,
                            "pred_tags": ner_pred_tags,
                        }
                    )
                    guid += 1
                    tokens = []
                    ner_true_tags = []
                    ner_pred_tags = []
            else:
                # splits = line.split("\t")
                splits = (
                    line.split("\t") if len(line.split("\t")) == 3 else line.split(" ")
                )
                tokens.append(splits[0].strip())
                ner_true_tags.append(splits[1].strip())
                ner_pred_tags.append(splits[2].strip())

        # last example
        data.append(
            {
                "id": str(guid),
                "tokens": tokens,
                "true_tags": ner_true_tags,
                "pred_tags": ner_pred_tags,
            }
        )
        return data
