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

    def test_sort_buckets_by_value(self):

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
            "dataset_name": "fb15k-237",
            "metric_names": ["Hits"],
            "sort_by": "value",
            "sort_by_metric": "first",
        }

        processor = get_processor(TaskType.kg_link_tail_prediction.value)

        sys_info = processor.process(metadata, data)

        # analysis.write_to_directory("./")
        self.assertIsNotNone(sys_info.results.fine_grained)
        self.assertGreater(len(sys_info.results.overall), 0)

        symmetry_performances = sys_info.results.fine_grained['symmetry']
        if len(symmetry_performances.values()) <= 1:  # can't sort if only 1 item
            return
        for i in range(len(symmetry_performances.values()) - 1):
            first_item = list(symmetry_performances.values())[i].performances[0].value
            second_item = (
                list(symmetry_performances.values())[i + 1].performances[0].value
            )
            # print('comparing:', first_item, second_item)
            self.assertGreater(first_item, second_item)

    def test_sort_buckets_by_key(self):

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
            "dataset_name": "fb15k-237",
            "metric_names": ["Hits"],
            "sort_by": "key",
        }

        processor = get_processor(TaskType.kg_link_tail_prediction.value)

        sys_info = processor.process(metadata, data)

        # analysis.write_to_directory("./")
        self.assertIsNotNone(sys_info.results.fine_grained)
        self.assertGreater(len(sys_info.results.overall), 0)

        symmetry_performances = sys_info.results.fine_grained['symmetry']
        if len(symmetry_performances.values()) <= 1:  # can't sort if only 1 item
            return
        for i in range(len(symmetry_performances.values()) - 1):
            first_item = list(symmetry_performances.values())[i].bucket_name
            second_item = list(symmetry_performances.values())[i + 1].bucket_name
            print('comparing:', first_item, second_item)
            self.assertGreater(second_item, first_item)


if __name__ == '__main__':
    unittest.main()
