import pathlib
import os
import unittest
from explainaboard import FileType, Source, TaskType, get_loader, get_processor

artifacts_path = os.path.dirname(pathlib.Path(__file__)) + "/artifacts/"


class TestNER(unittest.TestCase):
    def test_generate_system_analysis(self):
        """TODO: should add harder tests"""

        path_data = artifacts_path + "test-ner.tsv"
        loader = get_loader(
            TaskType.named_entity_recognition,
            path_data,
            Source.local_filesystem,
            FileType.conll,
        )
        data = loader.load()

        metadata = {
            "task_name": TaskType.named_entity_recognition.value,
            # "dataset_name": "conll2003",
            # "sub_dataset_name":"ner",
            "metric_names": ["f1_seqeval"],
        }

        processor = get_processor(TaskType.named_entity_recognition)
        # self.assertEqual(len(processor._features), 4)

        sys_info = processor.process(metadata, data)

        # analysis.write_to_directory("./")
        self.assertIsNotNone(sys_info.results.fine_grained)
        self.assertGreater(len(sys_info.results.overall), 0)


if __name__ == '__main__':
    unittest.main()
