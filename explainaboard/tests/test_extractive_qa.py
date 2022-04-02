import os
import pathlib
import unittest

from explainaboard import FileType, get_loader, get_processor, Source, TaskType

artifacts_path = os.path.dirname(pathlib.Path(__file__)) + "/artifacts/"


class TestExtractiveQA(unittest.TestCase):
    def test_extractive_qa_en(self):
        """TODO: should add harder tests"""

        path_data = artifacts_path + "test-xquad-en.json"
        loader = get_loader(
            TaskType.question_answering_extractive,
            path_data,
            Source.local_filesystem,
            FileType.json,
        )
        data = loader.load()

        metadata = {
            "task_name": TaskType.question_answering_extractive.value,
            "dataset_name": "squad",
            "metric_names": ["F1ScoreQA", "ExactMatchQA"],
            # "language":"en"
        }

        processor = get_processor(TaskType.question_answering_extractive)

        sys_info = processor.process(metadata, data)

        # analysis.write_to_directory("./")
        self.assertIsNotNone(sys_info.results.fine_grained)
        self.assertGreater(len(sys_info.results.overall), 0)
        # should be 0.6974789915966386
        self.assertAlmostEqual(
            sys_info.results.overall["ExactMatchQA"].value,
            0.6974789915966386,
            2,
            "almost equal",
        )
        # should be 0.8235975260931867
        self.assertAlmostEqual(
            sys_info.results.overall["F1ScoreQA"].value,
            0.8235975260931867,
            2,
            "almost equal",
        )

    def test_extractive_qa_zh(self):
        """TODO: should add harder tests"""

        path_data = artifacts_path + "test-xquad-zh.json"
        loader = get_loader(
            TaskType.question_answering_extractive,
            path_data,
            Source.local_filesystem,
            FileType.json,
        )
        data = loader.load()

        metadata = {
            "task_name": TaskType.question_answering_extractive.value,
            "dataset_name": "squad",
            "metric_names": ["F1ScoreQA", "ExactMatchQA"],
            "language": "zh",
        }

        processor = get_processor(TaskType.question_answering_extractive)

        sys_info = processor.process(metadata, data)

        # analysis.write_to_directory("./")
        self.assertIsNotNone(sys_info.results.fine_grained)
        self.assertGreater(len(sys_info.results.overall), 0)
        # 0.6285714285714286
        self.assertAlmostEqual(
            sys_info.results.overall["ExactMatchQA"].value,
            0.6285714285714286,
            2,
            "almost equal",
        )
        # 0.7559651817716333
        self.assertAlmostEqual(
            sys_info.results.overall["F1ScoreQA"].value,
            0.7559651817716333,
            2,
            "almost equal",
        )


if __name__ == '__main__':
    unittest.main()
