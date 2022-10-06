import os
import unittest

from integration_tests.utils import test_artifacts_path

from explainaboard import (
    FileType,
    get_loader_class,
    get_processor_class,
    Source,
    TaskType,
)


class MetaEvalWMTDATest(unittest.TestCase):
    artifact_path = os.path.join(test_artifacts_path, "meta_evaluation")
    tsv_dataset = os.path.join(artifact_path, "./wmt20-DA/cs-en/data.tsv")
    txt_output = os.path.join(artifact_path, "./wmt20-DA/cs-en/score.txt")

    def test_da_cs_en(self):

        metadata = {
            "task_name": TaskType.meta_evaluation_wmt_da.value,
            "metric_names": ["SysPearsonCorr"],
            "confidence_alpha": None,
        }
        loader = get_loader_class(TaskType.meta_evaluation_wmt_da)(
            self.tsv_dataset,
            self.txt_output,
            Source.local_filesystem,
            Source.local_filesystem,
            FileType.tsv,
            FileType.text,
        )
        data = loader.load().samples
        processor = get_processor_class(TaskType.meta_evaluation_wmt_da)()

        sys_info = processor.process(metadata, data)
        self.assertGreater(len(sys_info.results.analyses), 0)
        self.assertGreater(len(sys_info.results.overall), 0)
        self.assertAlmostEqual(
            sys_info.results.overall[0]["SegKtauCorr"].value, -0.0169, 3
        )
