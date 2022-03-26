import os
import pathlib
import unittest

from explainaboard import FileType, get_loader, get_processor, Source, TaskType

artifacts_path = os.path.dirname(pathlib.Path(__file__)) + "/artifacts/"


class TestTextPairClassification(unittest.TestCase):
    def test_snli(self):

        metadata = {
            "task_name": TaskType.text_classification.value,
            "metric_names": ["Accuracy"],
        }
        path_data = artifacts_path + "test-snli.tsv"
        loader = get_loader(
            TaskType.text_pair_classification,
            path_data,
            Source.local_filesystem,
            FileType.tsv,
        )
        data = list(loader.load())
        processor = get_processor(TaskType.text_pair_classification)

        sys_info = processor.process(metadata, data)

        # analysis.write_to_directory("./")
        self.assertIsNotNone(sys_info.results.fine_grained)
        self.assertGreater(len(sys_info.results.overall), 0)
