import pathlib
import os
import unittest
from explainaboard import FileType, Source, TaskType, get_loader, get_processor

artifacts_path = os.path.dirname(pathlib.Path(__file__)) + "/artifacts/"


class TestKgLinkTailPrediction(unittest.TestCase):
    def test_generate_system_analysis(self):

        path_data = artifacts_path + "test-kg-link-tail-prediction.json"
        loader = get_loader(
            TaskType.kg_link_tail_prediction,
            Source.local_filesystem,
            FileType.json,
            path_data,
        )
        data = loader.load()

        metadata = {
            "task_name": TaskType.kg_link_tail_prediction.value,
            "dataset_name": "fb15k-237-subset",
            "metric_names": ["Hits"],
        }

        processor = get_processor(
            TaskType.kg_link_tail_prediction.value
        )

        results = processor.process(metadata, data)

        # analysis.write_to_directory("./")
        self.assertIsNotNone(results.fine_grained)
        self.assertGreater(len(results.overall), 0)


if __name__ == '__main__':
    unittest.main()
