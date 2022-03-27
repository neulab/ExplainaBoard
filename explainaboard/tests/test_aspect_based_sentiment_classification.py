import os
import pathlib
import unittest

from explainaboard import FileType, get_loader, get_processor, Source, TaskType
from explainaboard.tests.utils import load_file_as_str

artifacts_path = os.path.dirname(pathlib.Path(__file__)) + "/artifacts/"


class TestAspectBasedSentimentClassification(unittest.TestCase):
    def test_e2e(self):
        metadata = {
            "task_name": TaskType.aspect_based_sentiment_classification.value,
            "metric_names": ["Accuracy", "F1Score"],
        }
        loader = get_loader(
            TaskType.aspect_based_sentiment_classification,
            load_file_as_str(f"{artifacts_path}test-aspect.tsv"),
            Source.in_memory,
            FileType.tsv,
        )
        data = loader.load()
        processor = get_processor(TaskType.aspect_based_sentiment_classification)

        sys_info = processor.process(metadata, data)

        # analysis.write_to_directory("./")
        self.assertIsNotNone(sys_info.results.fine_grained)
        self.assertGreater(len(sys_info.results.overall), 0)
