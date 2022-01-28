import pathlib
import os
import unittest
from explainaboard import FileType, Source, TaskType, get_loader, get_processor
from explainaboard.tests.utils import load_file_as_str

artifacts_path = os.path.dirname(pathlib.Path(__file__)) + "/artifacts/"

class TestTextClassification(unittest.TestCase):

    
    def test_e2e(self):

        metadata = {"task_name": TaskType.text_classification.value,
            "metric_names": ["Accuracy", "F1score"]}
        loader = get_loader(TaskType.text_classification, Source.in_memory, FileType.tsv,
                            load_file_as_str(f"{artifacts_path}sys_out1.tsv"))
        data = loader.load()
        processor = get_processor(TaskType.text_classification, metadata, data)




        analysis = processor.process()

        # analysis.write_to_directory("./")
        self.assertListEqual(analysis.metric_names, metadata["metric_names"])
        self.assertIsNotNone(analysis.results.fine_grained)
        self.assertGreater(len(analysis.results.overall), 0)

