from __future__ import annotations

import os
import unittest

from integration_tests.utils import test_artifacts_path

from explainaboard import FileType, get_processor_class, Source, TaskType
from explainaboard.loaders.file_loader import DatalabLoaderOption
from explainaboard.loaders.loader_factory import get_loader_class


class ClozeMultipleChoiceTest(unittest.TestCase):
    artifact_path = os.path.join(test_artifacts_path, "gaokao")
    json_output = os.path.join(artifact_path, "rst_2018_quanguojuan1_cloze_choice.json")

    def test_datalab_loader(self):
        loader = get_loader_class(TaskType.cloze_mutiple_choice).from_datalab(
            dataset=DatalabLoaderOption("gaokao2018_np1", "cloze-multiple-choice"),
            output_data=self.json_output,
            output_source=Source.local_filesystem,
            output_file_type=FileType.json,
        )
        data = loader.load()

        metadata = {
            "task_name": TaskType.cloze_mutiple_choice.value,
            "dataset_name": "gaokao2018_np1",
            "sub_dataset_name": "cloze-multiple-choice",
            "metric_names": ["CorrectCount"],
        }
        processor = get_processor_class(TaskType.cloze_mutiple_choice)()
        sys_info = processor.process(metadata, data, skip_failed_analyses=True)
        self.assertGreater(len(sys_info.results.analyses), 0)


if __name__ == "__main__":
    unittest.main()
