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
            Source.local_filesystem,
            FileType.conll,
            path_data,
        )
        data = loader.load()

        metadata = {
            "task_name": TaskType.named_entity_recognition.value,
            # "dataset_name": "conll2003",
            # "sub_dataset_name":"ner",
            "metric_names": ["f1_score_seqeval"],
        }

        processor = get_processor(TaskType.named_entity_recognition, metadata, data)
        # self.assertEqual(len(processor._features), 4)

        analysis = processor.process()
        analysis.write_to_directory("./")

        self.assertListEqual(analysis.metric_names, metadata["metric_names"])
        self.assertIsNotNone(analysis.results.fine_grained)
        self.assertGreater(len(analysis.results.overall), 0)


if __name__ == '__main__':
    unittest.main()
