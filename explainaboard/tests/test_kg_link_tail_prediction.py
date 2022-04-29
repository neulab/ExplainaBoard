import os
import unittest

from explainaboard import FileType, get_processor, TaskType
from explainaboard.loaders.loader_registry import get_custom_dataset_loader
from explainaboard.tests.utils import test_artifacts_path


class TestKgLinkTailPrediction(unittest.TestCase):
    artifact_path = os.path.join(test_artifacts_path, "kg_link_tail_prediction")
    dataset_no_custom_feature = os.path.join(artifact_path, "no_custom_feature.json")
    dataset_with_custom_feature = os.path.join(
        artifact_path, "with_custom_feature.json"
    )

    def test_no_user_defined_features(self):
        loader = get_custom_dataset_loader(
            TaskType.kg_link_tail_prediction,
            self.dataset_no_custom_feature,
            self.dataset_no_custom_feature,
            dataset_file_type=FileType.json,
            output_file_type=FileType.json,
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

        self.assertIsNotNone(sys_info.results.fine_grained)
        self.assertGreater(len(sys_info.results.overall), 0)

    def test_with_user_defined_features(self):
        loader = get_custom_dataset_loader(  # use defaults
            TaskType.kg_link_tail_prediction,
            self.dataset_with_custom_feature,
            self.dataset_with_custom_feature,
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
                "true_tail_anonymity",
                "true_label",
                "predictions",
                "rel_type",
            },
        )

    def test_sort_buckets_by_value(self):
        loader = get_custom_dataset_loader(
            TaskType.kg_link_tail_prediction,
            self.dataset_no_custom_feature,
            self.dataset_no_custom_feature,
        )
        data = loader.load()
        self.assertEqual(loader.user_defined_features_configs, {})

        metadata = {
            "task_name": TaskType.kg_link_tail_prediction.value,
            "dataset_name": "fb15k-237",
            "metric_names": ["Hits"],
            "sort_by": "performance_value",
            "sort_by_metric": "first",
        }

        processor = get_processor(TaskType.kg_link_tail_prediction.value)
        sys_info = processor.process(metadata, data)

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
            self.assertGreater(first_item, second_item)

    def test_sort_buckets_by_key(self):
        loader = get_custom_dataset_loader(
            TaskType.kg_link_tail_prediction,
            self.dataset_no_custom_feature,
            self.dataset_no_custom_feature,
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

        self.assertIsNotNone(sys_info.results.fine_grained)
        self.assertGreater(len(sys_info.results.overall), 0)

        symmetry_performances = sys_info.results.fine_grained['symmetry']
        if len(symmetry_performances.values()) <= 1:  # can't sort if only 1 item
            return
        for i in range(len(symmetry_performances.values()) - 1):
            first_item = list(symmetry_performances.values())[i].bucket_name
            second_item = list(symmetry_performances.values())[i + 1].bucket_name
            self.assertGreater(second_item, first_item)
