import json
import os
import unittest

from explainaboard import FileType, get_processor, Source, TaskType
from explainaboard.loaders.loader_registry import get_custom_dataset_loader
from explainaboard.tests.utils import test_artifacts_path


class TestSysDetails(unittest.TestCase):
    def test_generate_system_analysis(self):
        path_system_details = os.path.join(
            test_artifacts_path, "test_system_details.json"
        )
        dataset_data = os.path.join(
            test_artifacts_path, "text_classification", "dataset.tsv"
        )
        output_data = os.path.join(
            test_artifacts_path, "text_classification", "output.txt"
        )

        with open(path_system_details) as fin:
            system_details = json.load(fin)

        metadata = {
            "task_name": TaskType.text_classification,
            "metric_names": ["Accuracy"],
            "system_details": system_details,
        }

        loader = get_custom_dataset_loader(
            TaskType.text_classification,
            dataset_data,
            output_data,
            Source.local_filesystem,
            Source.local_filesystem,
            FileType.tsv,
            FileType.text,
        )
        data = loader.load()
        processor = get_processor(TaskType.text_classification)

        sys_info = processor.process(metadata, data)

        # analysis.write_to_directory("./")
        self.assertIsNotNone(
            sys_info.system_details, {"learning_rate": 0.0001, "number_of_layers": 10}
        )
