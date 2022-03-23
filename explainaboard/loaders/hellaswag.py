from typing import Dict, Iterable, List
from explainaboard.constants import FileType
from explainaboard.loaders.file_loader import TSVFileLoader
from .loader import register_loader
from .loader import Loader
from datasets import load_dataset
from explainaboard.tasks import TaskType


@register_loader(TaskType.hellaswag)
class HellaswagLoader(Loader):

    _default_file_type = FileType.tsv
    _default_file_loaders = {FileType.tsv: TSVFileLoader(None, False)}

    def load(self) -> Iterable[Dict]:
        """
        :param path_system_output: the path of system output file with following format:
        text \t label \t predicted_label
        :return: class object
        """
        dataset = load_dataset('hellaswag')['validation']

        data: List[Dict] = []
        if self._file_type == FileType.tsv:
            raw_data = self._default_file_loaders[FileType.tsv].load_raw(
                self._data, self._source
            )
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
