from __future__ import annotations

import os
import tempfile
import unittest

from integration_tests.utils import test_artifacts_path

from explainaboard import FileType, get_processor_class, TaskType
from explainaboard.analysis.analyses import BucketAnalysisDetails
from explainaboard.loaders.file_loader import FileLoaderMetadata
from explainaboard.loaders.loader_factory import get_loader_class
from explainaboard.metrics.metric import Score
from explainaboard.metrics.ranking import (
    HitsConfig,
    MeanRankConfig,
    MeanReciprocalRankConfig,
)
from explainaboard.utils.typing_utils import narrow, unwrap


class KgLinkTailPredictionTest(unittest.TestCase):
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
        loader = get_loader_class(task)(dataset, dataset)
        data = loader.load()
        # Initialize the processor and perform the processing
        processor = get_processor_class(TaskType.kg_link_tail_prediction)()
        sys_info = processor.process(
            metadata={}, sys_output=data.samples, skip_failed_analyses=True
        )

        with tempfile.TemporaryDirectory() as tempdir:
            # If you want to write out to disk you can use
            sys_info.write_to_directory(tempdir)

    def test_no_user_defined_features(self):
        loader = get_loader_class(TaskType.kg_link_tail_prediction)(
            self.test_data,
            self.dataset_no_custom_feature,
            dataset_file_type=FileType.json,
            output_file_type=FileType.json,
        )
        data = loader.load()
        self.assertEqual(data.metadata, FileLoaderMetadata())

        metadata = {
            "task_name": TaskType.kg_link_tail_prediction.value,
            "metric_configs": {
                "Hits4": HitsConfig(hits_k=4),  # you can modify k here
                "MRR": MeanReciprocalRankConfig(),
                "MR": MeanRankConfig(),
            },
        }

        processor = get_processor_class(TaskType.kg_link_tail_prediction)()

        sys_info = processor.process(metadata, data.samples, skip_failed_analyses=True)

        self.assertGreater(len(sys_info.results.analyses), 0)
        self.assertGreater(len(sys_info.results.overall), 0)

    def test_with_user_defined_features(self):
        loader = get_loader_class(TaskType.kg_link_tail_prediction)(
            # use defaults
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
                "true_head_decipher",
                "true_tail_decipher",
                "true_tail",
                "predict",
                "predictions",
                "rel_type",
                "true_rank",
            },
        )

    def test_sort_buckets_by_value(self):
        loader = get_loader_class(TaskType.kg_link_tail_prediction)(
            self.test_data,
            self.dataset_no_custom_feature,
        )
        data = loader.load()
        self.assertEqual(data.metadata, FileLoaderMetadata())

        metadata = {
            "task_name": TaskType.kg_link_tail_prediction.value,
            "dataset_name": "fb15k_237",
            "metric_configs": {
                "Hits4": HitsConfig(hits_k=4),
                "MRR": MeanReciprocalRankConfig(),
                "MR": MeanRankConfig(),
            },
            "sort_by": "performance_value",
            "sort_by_metric": "Hits4",
        }

        processor = get_processor_class(TaskType.kg_link_tail_prediction)()
        sys_info = processor.process(metadata, data.samples, skip_failed_analyses=True)

        self.assertGreater(len(sys_info.results.analyses), 0)
        self.assertGreater(len(sys_info.results.overall), 0)

        analysis_map = {x.name: x for x in sys_info.results.analyses if x is not None}
        symmetry_performances = narrow(
            BucketAnalysisDetails, analysis_map["symmetry"].details
        ).bucket_performances
        if len(symmetry_performances) <= 1:  # can't sort if only 1 item
            return
        for i in range(len(symmetry_performances) - 1):
            first_item = (
                symmetry_performances[i]
                .results["Hits4"]
                .get_value(Score, "score")
                .value
            )
            second_item = (
                symmetry_performances[i + 1]
                .results["Hits4"]
                .get_value(Score, "score")
                .value
            )
            self.assertGreater(first_item, second_item)

    def test_sort_buckets_by_key(self):
        loader = get_loader_class(TaskType.kg_link_tail_prediction)(
            self.test_data,
            self.dataset_no_custom_feature,
        )
        data = loader.load()
        self.assertEqual(data.metadata, FileLoaderMetadata())

        metadata = {
            "task_name": TaskType.kg_link_tail_prediction.value,
            "dataset_name": "fb15k_237",
            "metric_configs": {
                "Hits4": HitsConfig(hits_k=4),
                "MRR": MeanReciprocalRankConfig(),
                "MR": MeanRankConfig(),
            },
            "sort_by": "key",
        }

        processor = get_processor_class(TaskType.kg_link_tail_prediction)()

        sys_info = processor.process(metadata, data.samples, skip_failed_analyses=True)

        self.assertGreater(len(sys_info.results.analyses), 0)
        self.assertGreater(len(sys_info.results.overall), 0)

        analysis_map = {x.name: x for x in sys_info.results.analyses if x is not None}
        symmetry_performances = narrow(
            BucketAnalysisDetails, analysis_map["symmetry"].details
        ).bucket_performances
        if len(symmetry_performances) <= 1:  # can't sort if only 1 item
            return
        for i in range(len(symmetry_performances) - 1):
            first_item = unwrap(symmetry_performances[i].bucket_name)
            second_item = unwrap(symmetry_performances[i + 1].bucket_name)
            self.assertGreater(second_item, first_item)
