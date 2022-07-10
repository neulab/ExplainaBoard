import os
import unittest

from explainaboard import FileType, get_processor, Source, TaskType
from explainaboard.loaders.loader_registry import get_custom_dataset_loader
from explainaboard.tests.utils import test_artifacts_path


class TestTextRegression(unittest.TestCase):
    artifact_path = os.path.join(test_artifacts_path, "text_regression")
    tsv_dataset = os.path.join(artifact_path, "./wmt20-DA/cs-en/data.tsv")
    txt_output = os.path.join(artifact_path, "./wmt20-DA/cs-en/score.tsv")



    def test_da_cs_en(self):

        metadata = {
            "task_name": TaskType.text_regression.value,
            "metric_names": ["SysPearsonCorr"],
        }
        loader = get_custom_dataset_loader(
            TaskType.text_regression,
            self.tsv_dataset,
            self.txt_output,
            Source.local_filesystem,
            Source.local_filesystem,
            FileType.tsv,
            FileType.tsv,
        )
        data = loader.load()
        processor = get_processor(TaskType.text_regression)

        sys_info = processor.process(metadata, data)

        self.assertIsNotNone(sys_info.results.fine_grained)
        self.assertGreater(len(sys_info.results.overall), 0)
