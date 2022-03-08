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
        data = loader.load()

        metadata = {
            "task_name": TaskType.summarization.value,
            "dataset_name": "cnndm",
            "metric_names": ["bleu"],
        }

        processor = get_processor(TaskType.summarization.value, metadata, data)
        # self.assertEqual(len(processor._features), 4)

        analysis = processor.process()
        # analysis.write_to_directory("./")
        # print(analysis)
        self.assertListEqual(analysis.metric_names, metadata["metric_names"])
        self.assertIsNotNone(analysis.results.fine_grained)
        self.assertGreater(len(analysis.results.overall), 0)


if __name__ == '__main__':
    unittest.main()
