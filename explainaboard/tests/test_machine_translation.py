import pathlib
import os
import unittest
from explainaboard import FileType, Source, TaskType, get_loader, get_processor

artifacts_path = os.path.dirname(pathlib.Path(__file__)) + "/artifacts/"


class TestMachineTranslation(unittest.TestCase):
    def test_generate_system_analysis(self):
        """TODO: should add harder tests"""

        path_data = artifacts_path + "test-mt.tsv"
        loader = get_loader(
            TaskType.machine_translation,
            Source.local_filesystem,
            FileType.tsv,
            path_data,
        )
        data = loader.load()

        metadata = {
            "task_name": TaskType.machine_translation.value,
            "dataset_name": "ted_multi",
            "metric_names": ["bleu"],
        }

        processor = get_processor(TaskType.machine_translation.value)
        # self.assertEqual(len(processor._features), 4)

        sys_info = processor.process(metadata, data)

        # analysis.write_to_directory("./")
        self.assertIsNotNone(sys_info.results.fine_grained)
        self.assertGreater(len(sys_info.results.overall), 0)


if __name__ == '__main__':
    unittest.main()
