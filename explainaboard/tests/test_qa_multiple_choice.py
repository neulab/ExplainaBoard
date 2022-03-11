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
        data = loader.load()

        metadata = {
            "task_name": TaskType.qa_multiple_choice.value,
            "dataset_name": "metaphor_qa",
            "metric_names": ["Accuracy"],
        }

        processor = get_processor(TaskType.qa_multiple_choice.value, metadata, data)
        # self.assertEqual(len(processor._features), 4)

        analysis = processor.process()
        # analysis.write_to_directory("./")
        # print(analysis)
        self.assertListEqual(analysis.metric_names, metadata["metric_names"])
        self.assertIsNotNone(analysis.results.fine_grained)
        # self.assertGreater(len(analysis.results.overall), 1)


if __name__ == '__main__':
    unittest.main()
