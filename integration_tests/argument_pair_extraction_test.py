import os
import unittest

from integration_tests.utils import test_artifacts_path

from explainaboard import FileType, get_processor, Source, TaskType
from explainaboard.loaders.file_loader import DatalabLoaderOption
from explainaboard.loaders.loader_registry import get_loader_class


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
        processor = get_processor(TaskType.argument_pair_extraction)
        sys_info = processor.process(metadata, data)
        self.assertIsNotNone(sys_info.results.analyses)

        self.assertGreater(len(sys_info.results.overall), 0)
        self.assertAlmostEqual(
            sys_info.results.overall[0][0].value,
            0.25625,
            4,
            "almost equal",
        )


if __name__ == '__main__':
    unittest.main()
