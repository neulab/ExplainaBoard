import pathlib
import os
import unittest
from explainaboard import FileType, Source, TaskType, get_loader, get_processor

artifacts_path = os.path.dirname(pathlib.Path(__file__)) + "/artifacts/"


class TestKgLinkTailPrediction(unittest.TestCase):
    def test_no_user_defined_features(self):

        path_data = artifacts_path + "test-kg-link-tail-prediction.json"
        loader = get_loader(
            TaskType.kg_link_tail_prediction,
            Source.local_filesystem,
            FileType.json,
            path_data,
        )
        data = loader.load()
        self.assertEqual(loader.user_defined_features_configs, {})

        metadata = {
            "task_name": TaskType.kg_link_tail_prediction.value,
            "dataset_name": "fb15k-237-subset",
            "metric_names": ["Hits"],
        }

        processor = get_processor(TaskType.kg_link_tail_prediction.value)

        sys_info = processor.process(metadata, data)

        # analysis.write_to_directory("./")
        self.assertIsNotNone(sys_info.results.fine_grained)
        self.assertGreater(len(sys_info.results.overall), 0)

    def test_with_user_defined_features(self):
        loader = get_loader(
            TaskType.kg_link_tail_prediction,
            Source.local_filesystem,
            FileType.json,
            artifacts_path + "test-kg-link-tail-prediction-user-defined-features.json",
        )
        data = loader.load()
        self.assertEqual(len(loader.user_defined_features_configs), 2)
        self.assertEqual(len(data), 2)
        self.assertEqual(
            set(data[0].keys()),
            {
                "id",
                "link",
                "relation",
                "true_head",
                "true_tail",
                "predicted_tails",
                "user_defined_feature_1",
                "user_defined_feature_2",
            },
        )


if __name__ == '__main__':
    unittest.main()
