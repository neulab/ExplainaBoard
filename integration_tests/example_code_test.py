from __future__ import annotations

import tempfile
import unittest

from integration_tests.utils import top_path

from explainaboard import FileType, get_processor_class, Source, TaskType
from explainaboard.loaders import get_loader_class
from explainaboard.loaders.file_loader import DatalabLoaderOption


class ExampleCodeTest(unittest.TestCase):
    """
    This tests example code that is included in the documentation.
    """

    def test_readme_datalab_dataset(self):
        loader = get_loader_class(TaskType.text_classification).from_datalab(
            dataset=DatalabLoaderOption("sst2"),
            output_data=f"{top_path}/integration_tests/artifacts/text_classification/"
            "output_sst2.txt",
            output_source=Source.local_filesystem,
            output_file_type=FileType.text,
        )
        data = loader.load().samples
        processor = get_processor_class(TaskType.text_classification)()
        analysis = processor.process(
            metadata={}, sys_output=data, skip_failed_analyses=True
        )

        with tempfile.TemporaryDirectory() as tempdir:
            analysis.write_to_directory(tempdir)

    def test_readme_custom_dataset(self):
        dataset = f"{top_path}/integration_tests/artifacts/summarization/dataset.tsv"
        output = f"{top_path}/integration_tests/artifacts/summarization/output.txt"
        loader = get_loader_class(TaskType.summarization)(
            dataset_data=dataset, output_data=output
        )
        data = loader.load().samples
        processor = get_processor_class(TaskType.summarization)()
        analysis = processor.process(
            metadata={}, sys_output=data, skip_failed_analyses=True
        )

        with tempfile.TemporaryDirectory() as tempdir:
            analysis.write_to_directory(tempdir)
