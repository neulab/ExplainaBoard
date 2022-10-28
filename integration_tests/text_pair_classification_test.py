from __future__ import annotations

import os
import unittest

from integration_tests.utils import test_artifacts_path

from explainaboard import (
    DatalabLoaderOption,
    FileType,
    get_processor_class,
    Source,
    TaskType,
)
from explainaboard.loaders.loader_factory import get_loader_class


class TextPairClassificationTest(unittest.TestCase):
    artifact_path = os.path.join(test_artifacts_path, "text_pair_classification")
    tsv_dataset = os.path.join(artifact_path, "dataset-snli.tsv")
    txt_output = os.path.join(artifact_path, "output.txt")
    paws_fra_output = os.path.join(artifact_path, "paws_fra_output.txt")

    def test_load_tsv(self):
        loader = get_loader_class(TaskType.text_pair_classification)(
            self.tsv_dataset,
            self.txt_output,
            Source.local_filesystem,
            Source.local_filesystem,
            FileType.tsv,
            FileType.text,
        )
        data = loader.load()
        self.assertEqual(len(data), 5)
        self.assertEqual(
            data[1],
            {
                "text1": "This church choir sings to the masses as they sing joyous "
                + "songs from the book at a church.",
                "text2": "The church is filled with song.",
                "true_label": "entailment",
                "id": "1",
                "predicted_label": "entailment",
            },
        )

    def test_snli(self):

        metadata = {
            "task_name": TaskType.text_classification.value,
            "metric_names": ["Accuracy"],
        }
        loader = get_loader_class(TaskType.text_pair_classification)(
            self.tsv_dataset,
            self.txt_output,
            Source.local_filesystem,
            Source.local_filesystem,
            FileType.tsv,
            FileType.text,
        )
        data = loader.load()
        processor = get_processor_class(TaskType.text_pair_classification)()

        sys_info = processor.process(metadata, data.samples, skip_failed_analyses=True)

        self.assertGreater(len(sys_info.results.analyses), 0)
        self.assertGreater(len(sys_info.results.overall), 0)

    def test_paws_fra(self):

        metadata = {
            "task_name": TaskType.text_classification.value,
            "metric_names": ["Accuracy"],
        }
        loader = get_loader_class(TaskType.text_pair_classification).from_datalab(
            DatalabLoaderOption("xtreme", "PAWS-X.fra", split="test"),
            self.paws_fra_output,
            Source.local_filesystem,
            FileType.text,
        )
        data = loader.load()
        processor = get_processor_class(TaskType.text_pair_classification)()

        sys_info = processor.process(metadata, data.samples, skip_failed_analyses=True)

        self.assertGreater(len(sys_info.results.analyses), 0)
        self.assertGreater(len(sys_info.results.overall), 0)
