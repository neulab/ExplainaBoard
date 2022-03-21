from typing import Dict, Iterable, List
from explainaboard.constants import Source, FileType
from .loader import register_loader
from .loader import Loader
from explainaboard.tasks import TaskType


@register_loader(TaskType.conditional_generation)
@register_loader(TaskType.summarization)
@register_loader(TaskType.machine_translation)
class ConditionalGenerationLoader(Loader):
    """
    Validate and Reformat system output file with tsv format:
    text \t true_label \t predicted_label

    usage:
        please refer to `test_loaders.py`
    """

    def __init__(self, source: Source, file_type: FileType, data: str = None):

        if source is None:
            source = Source.local_filesystem
        if file_type is None:  # default format
            file_type = FileType.tsv

        self._source = source
        self._file_type = file_type
        self._data = data

    def load(self) -> Iterable[Dict]:
        """
        :param path_system_output: the path of system output file with following format:
        text \t label \t predicted_label
        :return: class object
        """

        super().load()
        data: List[Dict] = []
        if self._file_type == FileType.tsv:
            for id, dp in enumerate(self._raw_data):
                source, reference, hypothesis = dp[:3]
                data.append(
                    {
                        "id": str(id),
                        "source": source.strip(),
                        "reference": reference.strip(),
                        "hypothesis": hypothesis.strip(),
                    }
                )
        elif self._file_type == FileType.json:  # This function has been unittested
            for id, info in enumerate(self._raw_data):
                source, reference, hypothesis = (
                    info["source"],
                    info["references"],
                    info["hypothesis"],
                )
                data.append(
                    {
                        "id": str(id),
                        "source": source.strip(),
                        "reference": reference.strip(),
                        "hypothesis": hypothesis.strip(),
                    }
                )
        else:
            raise NotImplementedError
        return data
