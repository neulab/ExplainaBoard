import pathlib
import os
import unittest
from explainaboard import FileType, Source, TaskType, get_loader, get_processor
from explainaboard.tests.utils import load_file_as_str

artifacts_path = os.path.dirname(pathlib.Path(__file__)) + "/artifacts/"


class TestAspectBasedSentimentClassification(unittest.TestCase):
    _system_output_data = [
        {"id": 0, "aspect": "Boot time",
         "text": "Boot time  is super fast, around anywhere from 35 seconds to 1 minute.",
         "true_label": "positive", "predicted_label": "positive"},
        {"id": 3, "aspect": "Windows 8",
         "text": "Did not enjoy the new  Windows 8  and  touchscreen functions .",
         "true_label": "negative", "predicted_label": "negative"},
        {"id": 8, "aspect": "installation disk ( DVD )",
         "text": "No  installation disk (DVD)  is included.",
         "true_label": "neutral", "predicted_label": "negative"}
    ]

    def test_generate_system_analysis(self):
        """TODO: should add harder tests"""

        metadata = {"task_name": TaskType.aspect_based_sentiment_classification.value,
                    "metric_names": ["Accuracy", "F1score"]}

        processor = get_processor(TaskType.aspect_based_sentiment_classification.value, metadata, self._system_output_data)
        self.assertEqual(len(processor._features), 10)

        analysis = processor.process()
        self.assertListEqual(analysis.metric_names, metadata["metric_names"])
        self.assertIsNotNone(analysis.results.fine_grained)
        self.assertGreater(len(analysis.results.overall), 0)

    def test_e2e(self):
        metadata = {"task_name": TaskType.aspect_based_sentiment_classification.value,
                    "metric_names": ["Accuracy", "F1score"]}
        loader = get_loader(TaskType.aspect_based_sentiment_classification, Source.in_memory, FileType.tsv,
                            load_file_as_str(f"{artifacts_path}test-aspect.tsv"))
        data = loader.load()
        processor = get_processor(TaskType.aspect_based_sentiment_classification, metadata, data)
        self.assertEqual(len(processor._features), 10)

        analysis = processor.process()

        # analysis.write_to_directory("./")
        self.assertListEqual(analysis.metric_names, metadata["metric_names"])
        self.assertIsNotNone(analysis.results.fine_grained)
        self.assertGreater(len(analysis.results.overall), 0)

