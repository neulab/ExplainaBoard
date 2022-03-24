import pathlib
import os
import unittest
from explainaboard import FileType, Source, TaskType, get_loader, get_processor

artifacts_path = os.path.dirname(pathlib.Path(__file__)) + "/artifacts/"


class TestQAMultipleChoice(unittest.TestCase):
    def test_generate_system_analysis(self):
        """TODO: should add harder tests"""

        path_data = artifacts_path + "synthetic_metaphor_qa.json"
        loader = get_loader(
            TaskType.qa_multiple_choice,
            path_data,
            Source.local_filesystem,
            FileType.json,
        )
        data = list(loader.load())

        metadata = {
            "task_name": TaskType.qa_multiple_choice.value,
            "dataset_name": "metaphor_qa",
            "metric_names": ["Accuracy"],
        }

        processor = get_processor(TaskType.qa_multiple_choice.value)
        # self.assertEqual(len(processor._features), 4)

        sys_info = processor.process(metadata, data)

        # analysis.write_to_directory("./")
        self.assertIsNotNone(sys_info.results.fine_grained)
        self.assertGreater(len(sys_info.results.overall), 0)

    def test_multiple_qa_customized_feature(self):
        """TODO: should add harder tests"""

        path_data = artifacts_path + "test-metaphor-qa-customized-features.json"

        loader = get_loader(
            TaskType.qa_multiple_choice,
            path_data,
            Source.local_filesystem,
            FileType.json,
        )
        data = list(loader.load())

        metadata = {
            "task_name": TaskType.qa_multiple_choice.value,
            "dataset_name": "metaphor_qa",
            "metric_names": ["Accuracy"],
            "user_defined_features_configs": loader.user_defined_features_configs,  # don't forget this, otherwise the user-defined features will be ignored
        }

        processor = get_processor(TaskType.qa_multiple_choice.value)
        # self.assertEqual(len(processor._features), 4)

        sys_info = processor.process(metadata, data)

        # analysis.write_to_directory("./")
        self.assertIsNotNone(sys_info.results.fine_grained)
        self.assertGreater(len(sys_info.results.overall), 0)


if __name__ == '__main__':
    unittest.main()
