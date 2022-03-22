from typing import Dict, Iterable, List
from explainaboard.constants import FileType
from explainaboard.tasks import TaskType
from .loader import register_loader
from .loader import Loader


@register_loader(TaskType.aspect_based_sentiment_classification)
class AspectBasedSentimentClassificationLoader(Loader):
    """
    Validate and Reformat system output file with tsv format:
    aspect \t text \t true_label \t predicted_label

    usage:
        please refer to `test_loaders.py`
    """

    _default_file_type = FileType.tsv

    def load(self) -> Iterable[Dict]:
        """
        aspect \t text \t label \t predicted_label
        :return: class object
        """
        super().load()
        data: List[Dict] = []
        if self._file_type == FileType.tsv:
            for id, dp in enumerate(self._raw_data):
                aspect, text, true_label, predicted_label = dp[:4]
                data.append(
                    {
                        "id": str(id),
                        "aspect": aspect.strip(),
                        "text": text.strip(),
                        "true_label": true_label.strip(),
                        "predicted_label": predicted_label.strip(),
                    }
                )
        elif self._file_type == FileType.json:
            for id, info in enumerate(self._raw_data):
                aspect, text, true_label, predicted_label = (
                    info["aspect"],
                    info["text"],
                    info["true_label"],
                    info["predicted_label"],
                )
                data.append(
                    {
                        "id": str(id),
                        "aspect": aspect.strip,
                        "text": text.strip(),
                        "true_label": true_label.strip(),
                        "predicted_label": predicted_label.strip(),
                    }
                )
        else:
            raise NotImplementedError
        return data
