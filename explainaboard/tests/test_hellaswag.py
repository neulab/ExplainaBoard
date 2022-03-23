import os
import pathlib
from explainaboard import get_loader, TaskType, get_processor
import unittest

from explainaboard.constants import FileType, Source


class TestHellaSwag(unittest.TestCase):
    def test_loaders_and_processors(self):
        test_file_path = (
            os.path.dirname(pathlib.Path(__file__)) + "/artifacts/hellaswag.tsv"
        )

        loader = get_loader(
            TaskType.hellaswag,
            test_file_path,
            Source.local_filesystem,
            FileType.tsv,
        )
        data = loader.load()
        self.assertEqual(len(data), 20)
        self.assertIn("id", data[0])

        processor = get_processor(TaskType.hellaswag)
        report = processor.process({}, data)
        self.assertIsNotNone(report)
        self.assertGreater(len(report.results.overall), 0)
