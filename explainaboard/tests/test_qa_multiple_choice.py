import os
import unittest

from explainaboard import FileType, get_processor, Source, TaskType
from explainaboard.loaders.loader_registry import get_custom_dataset_loader
from explainaboard.tests.utils import test_artifacts_path


class TestQAMultipleChoice(unittest.TestCase):
    artifact_path = os.path.join(test_artifacts_path, "qa_multiple_choice")
    json_dataset = os.path.join(artifact_path, "dataset_synthetic_metaphor_qa.json")
    json_output = os.path.join(artifact_path, "output.json")

    def test_load_json(self):
        loader = get_custom_dataset_loader(
            TaskType.qa_multiple_choice,
            self.json_dataset,
            self.json_output,
            Source.local_filesystem,
            Source.local_filesystem,
            FileType.json,
            FileType.json,
        )
        data = loader.load()
        self.assertEqual(len(data), 4)
        self.assertEqual(
            data[0],
            {
                'context': 'The girl was as down-to-earth as a Michelin-starred canape',
                'question': '',
                'answers': {
                    'text': 'The girl was not down-to-earth at all.',
                    'option_index': 0,
                },
                'id': '0',
                'predicted_answers': {
                    'text': 'The girl was not down-to-earth at all.',
                    'option_index': 0,
                },
            },
        )

    def test_generate_system_analysis(self):
        loader = get_custom_dataset_loader(
            TaskType.qa_multiple_choice,
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
            "dataset_name": "metaphor_qa",
            "metric_names": ["Accuracy"],
        }

        processor = get_processor(TaskType.qa_multiple_choice.value)
        sys_info = processor.process(metadata, data)

        self.assertIsNotNone(sys_info.results.fine_grained)
        self.assertGreater(len(sys_info.results.overall), 0)

    def test_multiple_qa_customized_feature(self):
        dataset_path = os.path.join(self.artifact_path, "dataset_metaphor_qa.json")
        output_path = os.path.join(
            self.artifact_path, "output_metaphor_qa_customized_features.json"
        )
        loader = get_custom_dataset_loader(
            TaskType.qa_multiple_choice,
            dataset_path,
            output_path,
            Source.local_filesystem,
            Source.local_filesystem,
            FileType.json,
            FileType.json,
        )
        data = loader.load()
        self.assertEqual(data[0]["commonsense_category"], ["obj", "cul"])

        metadata = {
            "task_name": TaskType.qa_multiple_choice.value,
            "dataset_name": "metaphor_qa",
            "metric_names": ["Accuracy"],
            # don't forget this, otherwise the user-defined features will be ignored
            "user_defined_features_configs": loader.user_defined_features_configs,
        }

        processor = get_processor(TaskType.qa_multiple_choice.value)

        sys_info = processor.process(metadata, data)

        self.assertIsNotNone(sys_info.results.fine_grained)
        self.assertGreater(len(sys_info.results.overall), 0)


if __name__ == '__main__':
    unittest.main()
