from __future__ import annotations

import os
import unittest

from integration_tests.utils import test_artifacts_path

from explainaboard import FileType, get_processor_class, Source, TaskType
from explainaboard.loaders.file_loader import DatalabLoaderOption
from explainaboard.loaders.loader_factory import get_loader_class
from explainaboard.metrics.metric import Score


class ArgumentPairExtractionTest(unittest.TestCase):

    artifact_path = os.path.join(test_artifacts_path, "argument_pair_extraction")

    def test_datalab_loader(self):
        json_en_dataset = os.path.join(self.artifact_path, "ape_predictions.txt")

        loader = get_loader_class(TaskType.argument_pair_extraction).from_datalab(
            dataset=DatalabLoaderOption("ape"),
            output_data=json_en_dataset,
            output_source=Source.local_filesystem,
            output_file_type=FileType.conll,
        )
        data = loader.load()

        metadata = {
            "task_name": TaskType.argument_pair_extraction,
            "dataset_name": "ape",
            "metric_names": ["APEF1Score", "F1Score"],
        }
        processor = get_processor_class(TaskType.argument_pair_extraction)()
        sys_info = processor.process(metadata, data)
        self.assertGreater(len(sys_info.results.analyses), 0)

        overall = sys_info.results.overall["example"]
        self.assertGreater(len(overall), 0)
        self.assertAlmostEqual(
            overall["F1"].get_value(Score, "score").value, 0.25625, 4
        )


if __name__ == "__main__":
    unittest.main()
