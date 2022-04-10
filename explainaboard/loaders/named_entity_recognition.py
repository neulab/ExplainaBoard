from explainaboard.constants import FileType
from explainaboard.loaders.file_loader import (
    CoNLLFileLoader,
    FileLoader,
    FileLoaderField,
)
from explainaboard.loaders.loader import Loader, register_loader
from explainaboard.tasks import TaskType


@register_loader(TaskType.named_entity_recognition)
class NERLoader(Loader):
    """
    Validate and Reformat system output file with tsv format:
    token \t true_tag \t predicted_tag

    usage:
        please refer to `test_loaders.py`
    """

    @classmethod
    def default_file_type(cls) -> FileType:
        return FileType.conll

    @classmethod
    def default_dataset_file_loaders(cls) -> dict[FileType, FileLoader]:
        return {
            FileType.conll: CoNLLFileLoader(
                [
                    FileLoaderField(0, "tokens", str),
                    FileLoaderField(1, "true_tags", str),
                    FileLoaderField(2, "pred_tags", str),
                ]
            )
        }
