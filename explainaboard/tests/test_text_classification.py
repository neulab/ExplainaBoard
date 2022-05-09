import dataclasses
import os
import unittest

from explainaboard import FileType, get_processor, Source, TaskType
from explainaboard.loaders.file_loader import DatalabLoaderOption, FileLoaderMetadata
from explainaboard.loaders.loader_registry import (
    get_custom_dataset_loader,
    get_datalab_loader,
)
from explainaboard.tests.utils import load_file_as_str, test_artifacts_path


class TestTextClassification(unittest.TestCase):
    artifact_path = os.path.join(test_artifacts_path, "text_classification")
    tsv_dataset = os.path.join(artifact_path, "dataset.tsv")
    txt_output = os.path.join(artifact_path, "output.txt")
    json_dataset = os.path.join(artifact_path, "dataset.json")
    json_output = os.path.join(artifact_path, "output_user_metadata.json")

    def test_load_custom_dataset_tsv(self):
        loader = get_custom_dataset_loader(  # use defaults
            TaskType.text_classification,
            self.tsv_dataset,
            self.txt_output,
        )
        data = loader.load()
        self.assertEqual(len(data), 10)
        self.assertEqual(
            data[6],
            {
                "text": "a weird and wonderful comedy .",
                "true_label": "positive",
                "id": "6",
                "predicted_label": "positive",
            },
        )

    def test_load_custom_dataset_json(self):
        loader = get_custom_dataset_loader(
            TaskType.text_classification,
            self.json_dataset,
            self.json_output,
            dataset_file_type=FileType.json,
            output_file_type=FileType.json,
        )
        data = loader.load()
        self.assertNotEqual(data.metadata, FileLoaderMetadata())
        self.assertEqual(len(data), 7)
        self.assertEqual(
            data[6],
            {
                'text': 'guaranteed to move anyone who ever , , or rolled .',
                'true_label': 'positive',
                'id': '6',
                'predicted_label': 'positive',
            },
        )

    def test_load_dataset_from_datalab(self):
        loader = get_datalab_loader(
            TaskType.text_classification,
            dataset=DatalabLoaderOption("sst2"),
            output_data=os.path.join(self.artifact_path, "output_sst2.txt"),
            output_source=Source.local_filesystem,
            output_file_type=FileType.text,
        )
        data = loader.load()
        self.assertEqual(len(data), 1821)
        self.assertEqual(
            data[0],
            {
                'text': 'if you sometimes like to go to the movies to have fun , '
                'wasabi is a good place to start .',
                'true_label': 'positive',
                'id': '0',
                'predicted_label': 'positive',
            },
        )

    def test_process(self):
        metadata = {
            "task_name": TaskType.text_classification,
            "metric_names": ["Accuracy", "F1Score"],
        }
        loader = get_custom_dataset_loader(
            TaskType.text_classification,
            load_file_as_str(self.tsv_dataset),
            load_file_as_str(self.txt_output),
            Source.in_memory,
            Source.in_memory,
            FileType.tsv,
            FileType.text,
        )
        data = loader.load()
        processor = get_processor(TaskType.text_classification)
        sys_info = processor.process(metadata, data)

        self.assertIsNotNone(sys_info.results.fine_grained)
        self.assertGreater(len(sys_info.results.overall), 0)

    def test_process_training_set_dependent_features(self):
        metadata = {
            "task_name": TaskType.text_classification.value,
            "metric_names": ["Accuracy", "F1Score"],
            "dataset_name": "ag_news",
            "reload_stat": False,
        }
        loader = get_custom_dataset_loader(
            TaskType.text_classification,
            self.json_dataset,
            self.json_output,
            Source.local_filesystem,
            Source.local_filesystem,
            FileType.json,
            FileType.json,
        )
        data = loader.load()

        processor = get_processor(TaskType.text_classification)
        sys_info = processor.process(metadata, data)

        self.assertIsNotNone(sys_info.results.fine_grained)
        self.assertGreater(len(sys_info.results.overall), 0)

    def test_process_metadata_in_output_file(self):
        loader = get_custom_dataset_loader(
            TaskType.text_classification,
            self.json_dataset,
            self.json_output,
            Source.local_filesystem,
            Source.local_filesystem,
            FileType.json,
            FileType.json,
        )
        data = loader.load()
        self.assertNotEqual(data.metadata, FileLoaderMetadata)
        metadata = dataclasses.asdict(data.metadata)
        processor = get_processor(TaskType.text_classification)

        sys_info = processor.process(metadata, data.samples)

        self.assertIsNotNone(sys_info.results.fine_grained)
        self.assertGreater(len(sys_info.results.overall), 0)
