import os
import unittest

from explainaboard import FileType, get_processor, Source, TaskType
from explainaboard.loaders.loader_registry import get_custom_dataset_loader
from explainaboard.tests.utils import test_artifacts_path


class TestTextPairClassification(unittest.TestCase):
    artifact_path = os.path.join(test_artifacts_path, "text_pair_classification")
    tsv_dataset = os.path.join(artifact_path, "dataset-snli.tsv")
    txt_output = os.path.join(artifact_path, "output.txt")

    def test_load_tsv(self):
        loader = get_custom_dataset_loader(
            TaskType.text_pair_classification,
            self.tsv_dataset,
            self.txt_output,
            Source.local_filesystem,
            Source.local_filesystem,
            FileType.tsv,
            FileType.text,
        )
        data = loader.load()
        self.assertEqual(len(data), 5)
        self.assertEqual(
            data[1],
            {
                'text1': 'This church choir sings to the masses as they sing joyous '
                + 'songs from the book at a church.',
                'text2': 'The church is filled with song.',
                'true_label': 'entailment',
                'id': '1',
                'predicted_label': 'entailment',
            },
        )

    def test_snli(self):

        metadata = {
            "task_name": TaskType.text_classification.value,
            "metric_names": ["Accuracy"],
        }
        loader = get_custom_dataset_loader(
            TaskType.text_pair_classification,
            self.tsv_dataset,
            self.txt_output,
            Source.local_filesystem,
            Source.local_filesystem,
            FileType.tsv,
            FileType.text,
        )
        data = loader.load()
        processor = get_processor(TaskType.text_pair_classification)

        sys_info = processor.process(metadata, data)

        self.assertIsNotNone(sys_info.results.fine_grained)
        self.assertGreater(len(sys_info.results.overall), 0)
