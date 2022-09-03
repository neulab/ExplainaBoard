import os
import unittest

from integration_tests.utils import test_artifacts_path

from explainaboard import FileType, get_processor, Source, TaskType
from explainaboard.loaders.file_loader import FileLoaderMetadata
from explainaboard.loaders.loader_registry import get_loader_class
from explainaboard.metrics.ranking import HitsConfig


class KgLinkTailPredictionTest(unittest.TestCase):
    artifact_path = os.path.join(test_artifacts_path, "kg_link_tail_prediction")

    def test_no_user_defined_features(self):
        dataset = os.path.join(self.artifact_path, "no_custom_feature.json")
        loader = get_loader_class(TaskType.kg_link_tail_prediction)(
            dataset,
            dataset,
            Source.local_filesystem,
            Source.local_filesystem,
            FileType.json,
            FileType.json,
        )
        data = loader.load()
        self.assertEqual(data.metadata, FileLoaderMetadata())

        metadata = {
            "task_name": TaskType.kg_link_tail_prediction.value,
            "metric_configs": [HitsConfig(name='Hits4', hits_k=4)],
        }

        processor = get_processor(TaskType.kg_link_tail_prediction.value)

        sys_info = processor.process(metadata, data.samples, skip_failed_analyses=True)

        self.assertIsNotNone(sys_info.results.analyses)
        self.assertGreater(len(sys_info.results.overall), 0)
