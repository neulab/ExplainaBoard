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
            Source.local_filesystem,
            FileType.json,
            path_data,
        )
        data = list(loader.load())

        metadata = {
            "task_name": TaskType.qa_multiple_choice.value,
            "dataset_name": "metaphor_qa",
            "metric_names": ["Accuracy"],
        }

        processor = get_processor(TaskType.qa_multiple_choice.value)
        # self.assertEqual(len(processor._features), 4)

        results = processor.process(metadata, data)
        # analysis.write_to_directory("./")
        # print(analysis)
        self.assertIsNotNone(results.fine_grained)
        self.assertGreater(len(results.overall), 0)


if __name__ == '__main__':
    unittest.main()
