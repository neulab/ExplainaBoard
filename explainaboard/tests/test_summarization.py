import pathlib
import os
import unittest
from explainaboard import FileType, Source, TaskType, get_loader, get_processor

artifacts_path = os.path.dirname(pathlib.Path(__file__)) + "/artifacts/"


class TestSummarization(unittest.TestCase):
    def test_generate_system_analysis(self):
        """TODO: should add harder tests"""

        path_data = artifacts_path + "test-summ.tsv"
        loader = get_loader(
            TaskType.summarization, Source.local_filesystem, FileType.tsv, path_data
        )
        data = list(loader.load())

        metadata = {
            "task_name": TaskType.summarization.value,
            "dataset_name": "cnndm",
            "metric_names": ["bleu"],
        }

        processor = get_processor(TaskType.summarization.value)
        # self.assertEqual(len(processor._features), 4)

        results = processor.process(metadata, data)
        # analysis.write_to_directory("./")
        # print(analysis)
        self.assertIsNotNone(results.fine_grained)
        self.assertGreater(len(results.overall), 0)


if __name__ == '__main__':
    unittest.main()
