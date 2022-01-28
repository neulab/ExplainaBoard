import pathlib
import os
import unittest
from explainaboard import FileType, Source, TaskType, get_loader, get_processor


artifacts_path = os.path.dirname(pathlib.Path(__file__)) + "/artifacts/"

class TestTextPairClassification(unittest.TestCase):


    def test_snli(self):

        metadata = {"task_name": TaskType.text_classification.value,
            "metric_names": ["Accuracy"]}
        path_data = artifacts_path+ "test-snli.tsv"
        loader = get_loader(TaskType.text_pair_classification, Source.local_filesystem, FileType.tsv, path_data)
        data = loader.load()
        processor = get_processor(TaskType.text_pair_classification, metadata, data)
        self.assertEqual(len(processor._features), 8)

        analysis = processor.process()
        #analysis.to_memory()

        # analysis.write_to_directory("./")
        self.assertListEqual(analysis.metric_names, metadata["metric_names"])
        # self.assertIsNotNone(analysis.results.fine_grained)
        # self.assertGreater(len(analysis.results.overall), 0)

