import os
import pathlib
import unittest

from explainaboard import get_loader, get_processor, TaskType


class TestExampleCode(unittest.TestCase):
    """
    This tests example code that is included in the documentation.
    """

    def test_top_readme(self):
        """
        This tests the code in the top README.md file.
        """

        # The following code is not actually verbatim from the example
        artifacts_path = os.path.dirname(pathlib.Path(__file__)) + "/artifacts"
        path_data = f"{artifacts_path}/test-summ.tsv"
        # End non-verbatim code

        loader = get_loader(TaskType.summarization, data=path_data)
        data = list(loader.load())
        processor = get_processor(TaskType.summarization)
        analysis = processor.process(metadata={}, sys_output=data)
        analysis.write_to_directory("./")
