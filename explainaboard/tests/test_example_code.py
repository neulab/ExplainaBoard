import unittest

from explainaboard import FileType, get_processor, Source, TaskType
from explainaboard.loaders.file_loader import DatalabLoaderOption
from explainaboard.loaders.loader_registry import get_loader, get_loader_custom_dataset


class TestExampleCode(unittest.TestCase):
    """
    This tests example code that is included in the documentation.
    """

    def test_readme_datalab_dataset(self):
        loader = get_loader(
            TaskType.text_classification,
            dataset=DatalabLoaderOption("sst2"),
            output_data="./explainaboard/tests/artifacts/text_classification/"
            "output_sst2.txt",
            output_source=Source.local_filesystem,
            output_file_type=FileType.text,
        )
        data = loader.load()
        processor = get_processor(TaskType.text_classification)
        analysis = processor.process(metadata={}, sys_output=data)
        analysis.write_to_directory("./")

    def test_readme_custom_dataset(self):
        dataset = "./explainaboard/tests/artifacts/summarization/dataset.tsv"
        output = "./explainaboard/tests/artifacts/summarization/output.txt"
        loader = get_loader_custom_dataset(
            TaskType.summarization, dataset_data=dataset, output_data=output
        )
        data = loader.load()
        processor = get_processor(TaskType.summarization)
        analysis = processor.process(metadata={}, sys_output=data)
        analysis.write_to_directory("./")
