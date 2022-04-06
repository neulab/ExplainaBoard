import os
import pathlib
import unittest

from explainaboard import FileType, get_loader, get_processor, Source, TaskType

artifacts_path = os.path.dirname(pathlib.Path(__file__)) + "/artifacts/"


class TestKgLinkTailPrediction(unittest.TestCase):
    def test_no_user_defined_features(self):

        path_data = artifacts_path + "test-kg-prediction-no-user-defined-new.json"
        loader = get_loader(
            TaskType.kg_link_tail_prediction,
            path_data,
            Source.local_filesystem,
            FileType.json,
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
            artifacts_path + "test-kg-prediction-user-defined-new.json",
            Source.local_filesystem,
            FileType.json,
        )
        data = loader.load()
        self.assertEqual(len(loader.user_defined_features_configs), 1)
        self.assertEqual(len(data), 10)
        self.assertEqual(
            set(data[0].keys()),
            {
                "id",
                "true_head",
                "true_link",
                "true_tail",
                "true_label",
                "predictions",
                "rel_type",
            },
        )


if __name__ == '__main__':
    unittest.main()
