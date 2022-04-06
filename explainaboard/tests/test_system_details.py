import json
import os
import pathlib
import unittest

from explainaboard import FileType, get_loader, get_processor, Source, TaskType

artifacts_path = os.path.dirname(pathlib.Path(__file__)) + "/artifacts/"


class TestSysDetails(unittest.TestCase):
    def test_generate_system_analysis(self):
        """TODO: should add harder tests"""

        path_system_details = artifacts_path + "test_system_details.json"
        path_data = artifacts_path + "sys_out1.tsv"

        with open(path_system_details) as fin:
            system_details = json.load(fin)

        metadata = {
            "task_name": TaskType.text_classification.value,
            "metric_names": ["Accuracy"],
            "system_details": system_details,
        }

        loader = get_loader(
            TaskType.text_classification,
            path_data,
            Source.local_filesystem,
            FileType.tsv,
        )
        data = list(loader.load())
        processor = get_processor(TaskType.text_classification)

        sys_info = processor.process(metadata, data)

        # analysis.write_to_directory("./")
        self.assertIsNotNone(
            sys_info.system_details, {"learning_rate": 0.0001, "number_of_layers": 10}
        )


if __name__ == '__main__':
    unittest.main()
