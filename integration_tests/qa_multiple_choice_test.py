from __future__ import annotations

import os
import unittest

from integration_tests.utils import test_artifacts_path

from explainaboard import FileType, get_processor_class, Source, TaskType
from explainaboard.loaders.loader_factory import get_loader_class


class QAMultipleChoiceTest(unittest.TestCase):
    artifact_path = os.path.join(test_artifacts_path, "qa_multiple_choice")
    json_dataset = os.path.join(artifact_path, "dataset_synthetic_fig_qa.json")
    json_output = os.path.join(artifact_path, "output.json")

    def test_load_json(self):
        loader = get_loader_class(TaskType.qa_multiple_choice)(
            self.json_dataset,
            self.json_output,
            Source.local_filesystem,
            Source.local_filesystem,
            FileType.json,
            FileType.json,
        )
        data = loader.load()
        self.assertEqual(len(data), 4)

    def test_generate_system_analysis(self):
        loader = get_loader_class(TaskType.qa_multiple_choice)(
            self.json_dataset,
            self.json_output,
            Source.local_filesystem,
            Source.local_filesystem,
            FileType.json,
            FileType.json,
        )
        data = loader.load()
        metadata = {
            "task_name": TaskType.qa_multiple_choice.value,
            "dataset_name": "fig_qa",
            "metric_names": ["Accuracy"],
        }

        processor = get_processor_class(TaskType.qa_multiple_choice)()
        sys_info = processor.process(metadata, data)

        self.assertGreater(len(sys_info.results.analyses), 0)
        self.assertGreater(len(sys_info.results.overall), 0)

    def test_multiple_qa_customized_feature(self):
        dataset_path = os.path.join(self.artifact_path, "dataset_fig_qa.json")
        output_path = os.path.join(
            self.artifact_path, "output_fig_qa_customized_features.json"
        )
        loader = get_loader_class(TaskType.qa_multiple_choice)(
            dataset_path,
            output_path,
            Source.local_filesystem,
            Source.local_filesystem,
            FileType.json,
            FileType.json,
        )
        data = loader.load()
        self.assertIsInstance(data.samples[0]["commonsense_category"], list)
        self.assertEqual(data.samples[0]["commonsense_category"], ["obj", "cul"])

        metadata = {
            "task_name": TaskType.qa_multiple_choice.value,
            "dataset_name": "fig_qa",
            "metric_names": ["Accuracy"],
            # don't forget this, otherwise the user-defined features will be ignored
            "custom_features": data.metadata.custom_features,
        }

        processor = get_processor_class(TaskType.qa_multiple_choice)()
        sys_info = processor.process(metadata, data.samples)

        self.assertEqual(len(sys_info.results.analyses), 5)
        self.assertGreater(len(sys_info.results.overall), 0)


if __name__ == "__main__":
    unittest.main()
