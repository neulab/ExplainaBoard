from explainaboard.loaders.file_loader import CoNLLFileLoader

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
    _default_file_loaders = {FileType.conll: CoNLLFileLoader()}
