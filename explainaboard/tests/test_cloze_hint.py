import os
import unittest

from explainaboard import FileType, get_processor, Source, TaskType
from explainaboard.loaders.file_loader import DatalabLoaderOption
from explainaboard.loaders.loader_registry import get_datalab_loader
from explainaboard.tests.utils import test_artifacts_path


class TestClozeGenerative(unittest.TestCase):
    artifact_path = os.path.join(test_artifacts_path, "gaokao")
    json_output = os.path.join(artifact_path, "rst_2018_quanguojuan1_cloze_hint.json")

    def test_datalab_loader(self):
        loader = get_datalab_loader(
            TaskType.cloze_generative,
            dataset=DatalabLoaderOption("gaokao2018_np1", "cloze-hint"),
            output_data=self.json_output,
            output_source=Source.local_filesystem,
            output_file_type=FileType.json,
        )
        data = loader.load()

        metadata = {
            "task_name": TaskType.cloze_generative.value,
            "dataset_name": "gaokao2018_np1",
            "sub_dataset_name": "cloze-hint",
            "metric_names": ["CorrectCount"],
        }
        processor = get_processor(TaskType.cloze_generative.value)
        sys_info = processor.process(metadata, data)
        self.assertIsNotNone(sys_info.results.fine_grained)


if __name__ == '__main__':
    unittest.main()
