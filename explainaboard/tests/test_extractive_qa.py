import pathlib
import os
import unittest
from explainaboard import FileType, Source, TaskType, get_loader, get_processor

artifacts_path = os.path.dirname(pathlib.Path(__file__)) + "/artifacts/"


class TestExtractiveQA(unittest.TestCase):
    def test_generate_system_analysis(self):
        """TODO: should add harder tests"""

        path_data = artifacts_path + "test-qa-squad.json"
        loader = get_loader(
            TaskType.question_answering_extractive,
            Source.local_filesystem,
            FileType.json,
            path_data,
        )
        data = loader.load()

        metadata = {
            "task_name": TaskType.question_answering_extractive.value,
            "dataset_name": "squad",
            "metric_names": ["f1_score_qa", "exact_match_qa"],
        }

        processor = get_processor(
            TaskType.question_answering_extractive, metadata, data
        )
        # self.assertEqual(len(processor._features), 4)

        analysis = processor.process()
        # analysis.write_to_directory("./")
        # print(analysis)
        self.assertIsNotNone(analysis.results.fine_grained)
        self.assertGreater(len(analysis.results.overall), 0)


if __name__ == '__main__':
    unittest.main()
