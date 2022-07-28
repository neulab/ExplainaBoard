import unittest

from explainaboard import FileType, get_processor, Source, TaskType
from explainaboard.loaders import get_loader_class
from explainaboard.loaders.file_loader import DatalabLoaderOption
from explainaboard.tests.utils import top_path


class TestExampleCode(unittest.TestCase):
    """
    This tests example code that is included in the documentation.
    """

    def test_readme_datalab_dataset(self):
        loader = get_loader_class(TaskType.text_classification).from_datalab(
            dataset=DatalabLoaderOption("sst2"),
            output_data=f"{top_path}/explainaboard/tests/artifacts/text_classification/"
            "output_sst2.txt",
            output_source=Source.local_filesystem,
            output_file_type=FileType.text,
        )
        data = loader.load()
        processor = get_processor(TaskType.text_classification)
        analysis = processor.process(metadata={}, sys_output=data)
        analysis.write_to_directory("./")

    def test_readme_custom_dataset(self):
        dataset = f"{top_path}/explainaboard/tests/artifacts/summarization/dataset.tsv"
        output = f"{top_path}/explainaboard/tests/artifacts/summarization/output.txt"
        loader = get_loader_class(TaskType.summarization)(
            dataset_data=dataset, output_data=output
        )
        data = loader.load()
        processor = get_processor(TaskType.summarization)
        analysis = processor.process(metadata={}, sys_output=data)
        analysis.write_to_directory("./")
