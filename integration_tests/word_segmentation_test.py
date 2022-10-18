from __future__ import annotations

import os
import unittest

from integration_tests.utils import test_artifacts_path

from explainaboard import FileType, get_processor_class, Source, TaskType
from explainaboard.loaders.loader_factory import get_loader_class


class WordSegmentationTest(unittest.TestCase):
    artifact_path = os.path.join(test_artifacts_path, "cws")
    conll_dataset = os.path.join(artifact_path, "test.tsv")
    conll_output = os.path.join(artifact_path, "prediction.tsv")

    def test_generate_system_analysis(self):
        loader = get_loader_class(TaskType.word_segmentation)(
            self.conll_dataset,
            self.conll_output,
            Source.local_filesystem,
            Source.local_filesystem,
            FileType.conll,
            FileType.conll,
        )
        data = loader.load()

        metadata = {
            "task_name": TaskType.word_segmentation.value,
            # "dataset_name": "conll2003",
            # "sub_dataset_name":"ner",
            "metric_names": ["F1Score"],
        }

        processor = get_processor_class(TaskType.word_segmentation)()

        sys_info = processor.process(metadata, data, skip_failed_analyses=True)

        self.assertGreater(len(sys_info.results.analyses), 0)
        self.assertGreater(len(sys_info.results.overall), 0)
