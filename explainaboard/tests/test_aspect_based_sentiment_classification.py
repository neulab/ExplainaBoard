import os
import unittest

from explainaboard import FileType, get_processor, Source, TaskType
from explainaboard.loaders.loader_registry import get_custom_dataset_loader
from explainaboard.tests.utils import load_file_as_str, test_artifacts_path


class TestAspectBasedSentimentClassification(unittest.TestCase):
    artifact_path = os.path.join(
        test_artifacts_path, "aspect_based_sentiment_classification"
    )
    tsv_dataset = os.path.join(artifact_path, "dataset.tsv")
    txt_output = load_file_as_str(os.path.join(artifact_path, "output.txt"))

    def test_e2e(self):
        loader = get_custom_dataset_loader(
            TaskType.aspect_based_sentiment_classification,
            self.tsv_dataset,
            self.txt_output,
            Source.local_filesystem,
            Source.in_memory,
            FileType.tsv,
            FileType.text,
        )
        data = loader.load()
        self.assertEqual(len(data), 100)
        self.assertEqual(
            data[0],
            {
                'aspect': 'Boot time',
                'text': 'Boot time  is super fast, around anywhere from 35 seconds to '
                + '1 minute.',
                'true_label': 'positive',
                'id': '0',
                'predicted_label': 'positive',
            },
        )

        metadata = {
            "task_name": TaskType.aspect_based_sentiment_classification,
            "metric_names": ["Accuracy", "F1Score"],
        }
        processor = get_processor(TaskType.aspect_based_sentiment_classification)

        sys_info = processor.process(metadata, data)

        self.assertIsNotNone(sys_info.results.fine_grained)
        self.assertGreater(len(sys_info.results.overall), 0)
