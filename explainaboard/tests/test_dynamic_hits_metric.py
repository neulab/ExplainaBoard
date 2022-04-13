import os
import unittest

from explainaboard import FileType, get_processor, Source, TaskType
from explainaboard.loaders.loader_registry import get_custom_dataset_loader
from explainaboard.metric import HitsConfig
from explainaboard.tests.utils import test_artifacts_path


class TestKgLinkTailPrediction(unittest.TestCase):
    artifact_path = os.path.join(test_artifacts_path, "kg_link_tail_prediction")

    def test_no_user_defined_features(self):
        dataset = os.path.join(self.artifact_path, "no_custom_feature.json")
        loader = get_custom_dataset_loader(
            TaskType.kg_link_tail_prediction,
            dataset,
            dataset,
            Source.local_filesystem,
            Source.local_filesystem,
            FileType.json,
            FileType.json,
        )
        data = loader.load()
        self.assertEqual(loader.user_defined_features_configs, {})

        metadata = {
            "task_name": TaskType.kg_link_tail_prediction.value,
            "dataset_name": "fb15k-237-subset",
            "metric_names": ["Hits"],
            "metric_configs": {"Hits": HitsConfig(hits_k=4)},
        }

        processor = get_processor(TaskType.kg_link_tail_prediction.value)

        sys_info = processor.process(metadata, data)

        self.assertIsNotNone(sys_info.results.fine_grained)
        self.assertGreater(len(sys_info.results.overall), 0)
