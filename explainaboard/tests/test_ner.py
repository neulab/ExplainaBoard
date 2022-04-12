import os
import unittest

from explainaboard import FileType, get_processor, Source, TaskType
from explainaboard.loaders.loader_registry import get_custom_dataset_loader
from explainaboard.tests.utils import test_artifacts_path


class TestNER(unittest.TestCase):
    artifact_path = os.path.join(test_artifacts_path, "ner")
    conll_dataset = os.path.join(artifact_path, "dataset.tsv")
    conll_output = os.path.join(artifact_path, "output.tsv")

    def test_generate_system_analysis(self):
        loader = get_custom_dataset_loader(
            TaskType.named_entity_recognition,
            self.conll_dataset,
            self.conll_output,
            Source.local_filesystem,
            Source.local_filesystem,
            FileType.conll,
            FileType.conll,
        )
        data = loader.load()
        self.assertEqual(len(data), 4)

        self.assertEqual(
            data[0]["tokens"],
            [
                "SOCCER",
                "-",
                "JAPAN",
                "GET",
                "LUCKY",
                "WIN",
                ",",
                "CHINA",
                "IN",
                "SURPRISE",
                "DEFEAT",
                ".",
            ],
        )
        self.assertEqual(data[1]["pred_tags"], ["B-PER", "I-PER"])

        metadata = {
            "task_name": TaskType.named_entity_recognition.value,
            # "dataset_name": "conll2003",
            # "sub_dataset_name":"ner",
            "metric_names": ["F1Score"],
        }

        processor = get_processor(TaskType.named_entity_recognition)

        sys_info = processor.process(metadata, data)

        self.assertIsNotNone(sys_info.results.fine_grained)
        self.assertGreater(len(sys_info.results.overall), 0)
