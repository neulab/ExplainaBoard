import pathlib
import os
import unittest
from explainaboard import FileType, Source, TaskType, get_loader, get_processor

artifacts_path = os.path.dirname(pathlib.Path(__file__)) + "/artifacts/"
class TestKgLinkTailPrediction(unittest.TestCase):

    
    def test_generate_system_analysis(self):
        
        path_data = artifacts_path + "test-kg-link-tail-prediction.json"
        loader = get_loader(TaskType.kg_link_tail_prediction, Source.local_filesystem, FileType.json, path_data)
        data = loader.load()

        metadata = {
            "task_name": TaskType.kg_link_tail_prediction.value,
            "dataset_name": "fb15k-237",
            "metric_names": ["Hits"]
        }
        
        processor = get_processor(TaskType.kg_link_tail_prediction.value, metadata, data)

        analysis = processor.process()

        # analysis.write_to_directory("./")
        self.assertListEqual(analysis.metric_names, metadata["metric_names"])
        self.assertIsNotNone(analysis.results.fine_grained)
        self.assertGreater(len(analysis.results.overall), 0)


if __name__ == '__main__':
    unittest.main()