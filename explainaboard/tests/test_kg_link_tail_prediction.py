import os
import unittest

from explainaboard import FileType, get_processor, TaskType
from explainaboard.loaders.file_loader import FileLoaderMetadata
from explainaboard.loaders.loader_registry import get_custom_dataset_loader
from explainaboard.metrics.ranking import (
    HitsConfig,
    MeanRankConfig,
    MeanReciprocalRankConfig,
)
from explainaboard.tests.utils import test_artifacts_path


class TestKgLinkTailPrediction(unittest.TestCase):
    artifact_path = os.path.join(test_artifacts_path, "kg_link_tail_prediction")
    test_data = os.path.join(artifact_path, "data_mini.json")
    dataset_no_custom_feature = os.path.join(artifact_path, "no_custom_feature.json")
    dataset_with_custom_feature = os.path.join(
        artifact_path, "with_custom_feature.json"
    )

    def test_simple_example(self):
        # Load the data
        dataset = self.dataset_no_custom_feature
        task = TaskType.kg_link_tail_prediction
        loader = get_custom_dataset_loader(task, dataset, dataset)
        data = loader.load()
        # Initialize the processor and perform the processing
        processor = get_processor(TaskType.kg_link_tail_prediction.value)
        sys_info = processor.process(metadata={}, sys_output=data.samples)
        # If you want to write out to disk you can use
        sys_info.write_to_directory('./')

    def test_no_user_defined_features(self):
        loader = get_custom_dataset_loader(
            TaskType.kg_link_tail_prediction,
            self.test_data,
            self.dataset_no_custom_feature,
            dataset_file_type=FileType.json,
            output_file_type=FileType.json,
        )
        data = loader.load()
        self.assertEqual(data.metadata, FileLoaderMetadata())

        metadata = {
            "task_name": TaskType.kg_link_tail_prediction.value,
            "dataset_name": "fb15k-237-subset",
            "metric_configs": [
                HitsConfig(name='Hits4', hits_k=4),  # you can modify k here
                MeanReciprocalRankConfig(name='MRR'),
                MeanRankConfig(name='MR'),
            ],
        }

        processor = get_processor(TaskType.kg_link_tail_prediction.value)

        sys_info = processor.process(metadata, data.samples)

        self.assertIsNotNone(sys_info.results.fine_grained)
        self.assertGreater(len(sys_info.results.overall), 0)

    def test_with_user_defined_features(self):
        loader = get_custom_dataset_loader(  # use defaults
            TaskType.kg_link_tail_prediction,
            self.test_data,
            self.dataset_with_custom_feature,
        )
        data = loader.load()
        self.assertEqual(len(data.metadata.custom_features), 1)
        self.assertEqual(len(data), 10)
        self.assertEqual(
            set(data[0].keys()),
            {
                "id",
                "true_head",
                "true_link",
                'true_head_decipher',
                'true_tail_decipher',
                "true_tail",
                "predict",
                "predictions",
                "rel_type",
                "true_rank",
            },
        )

    def test_sort_buckets_by_value(self):
        loader = get_custom_dataset_loader(
            TaskType.kg_link_tail_prediction,
            self.test_data,
            self.dataset_no_custom_feature,
        )
        data = loader.load()
        self.assertEqual(data.metadata, FileLoaderMetadata())

        metadata = {
            "task_name": TaskType.kg_link_tail_prediction.value,
            "dataset_name": "fb15k-237",
            "metric_configs": [
                HitsConfig(name='Hits4', hits_k=4),
                MeanReciprocalRankConfig(name='MRR'),
                MeanRankConfig(name='MR'),
            ],
            "sort_by": "performance_value",
            "sort_by_metric": "first",
        }

        processor = get_processor(TaskType.kg_link_tail_prediction.value)
        sys_info = processor.process(metadata, data.samples)

        self.assertIsNotNone(sys_info.results.fine_grained)
        self.assertGreater(len(sys_info.results.overall), 0)

        symmetry_performances = sys_info.results.fine_grained['symmetry']
        if len(symmetry_performances) <= 1:  # can't sort if only 1 item
            return
        for i in range(len(symmetry_performances) - 1):
            first_item = symmetry_performances[i].performances[0].value
            second_item = symmetry_performances[i + 1].performances[0].value
            self.assertGreater(first_item, second_item)

    def test_sort_buckets_by_key(self):
        loader = get_custom_dataset_loader(
            TaskType.kg_link_tail_prediction,
            self.test_data,
            self.dataset_no_custom_feature,
        )
        data = loader.load()
        self.assertEqual(data.metadata, FileLoaderMetadata())

        metadata = {
            "task_name": TaskType.kg_link_tail_prediction.value,
            "dataset_name": "fb15k-237",
            "metric_configs": [
                HitsConfig(name='Hits4', hits_k=4),
                MeanReciprocalRankConfig(name='MRR'),
                MeanRankConfig(name='MR'),
            ],
            "sort_by": "key",
        }

        processor = get_processor(TaskType.kg_link_tail_prediction.value)

        sys_info = processor.process(metadata, data.samples)

        self.assertIsNotNone(sys_info.results.fine_grained)
        self.assertGreater(len(sys_info.results.overall), 0)

        symmetry_performances = sys_info.results.fine_grained['symmetry']
        if len(symmetry_performances) <= 1:  # can't sort if only 1 item
            return
        for i in range(len(symmetry_performances) - 1):
            first_item = symmetry_performances[i].bucket_interval
            second_item = symmetry_performances[i + 1].bucket_interval
            self.assertGreater(second_item, first_item)
